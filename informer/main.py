import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import time
import math
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. 配置参数 (Config)
# ==========================================
class Config:
    def __init__(self):
        # 数据集路径
        self.data_path = 'ETTh1.csv'
        self.target = 'OT'  # 预测目标列

        # 任务参数
        self.seq_len = 96  # 输入序列长度
        self.label_len = 48  # Start token 长度 (通常为 seq_len 的一半)
        self.pred_len = 96  # 预测序列长度

        # 模型参数
        self.enc_in = 7  # Encoder 输入特征数 (ETTh1通常是7列)
        self.dec_in = 7  # Decoder 输入特征数
        self.c_out = 7  # 输出特征数
        self.d_model = 512  # 模型维度
        self.n_heads = 8  # 多头注意力头数
        self.e_layers = 2  # Encoder 层数
        self.d_layers = 1  # Decoder 层数
        self.d_ff = 2048  # FFN 维度
        self.factor = 5  # ProbSparse Attention 的采样因子
        self.dropout = 0.05
        self.attn = 'prob'  # 'prob' for ProbSparse, 'full' for Full Attention
        self.embed = 'timeF'  # 时间特征编码方式
        self.activation = 'gelu'
        self.distil = True  # 是否使用卷积蒸馏
        self.output_attention = False

        # 训练参数
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.train_epochs = 10
        self.patience = 3
        self.num_workers = 0
        self.checkpoints = './checkpoints/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = Config()


# ==========================================
# 2. 数据处理与加载 (Data Loading)
# ==========================================
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, dates, freq='h'):
        dates = pd.to_datetime(dates.values)
        return np.vstack([
            dates.month,
            dates.day,
            dates.weekday,
            dates.hour
        ]).transpose()


class Dataset_ETTh1(Dataset):
    def __init__(self, root_path='.', data_path='ETTh1.csv', flag='train',
                 size=None, features='M', target='OT', scale=True):
        # size: [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # Initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ETTh1 数据切分: Train: 12个月, Val: 4个月, Test: 4个月
        # 总数据量 approx 17420.
        # Train: 0-12*30*24 = 8640 (大致)
        # 这里使用标准的分割边界
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # 时间特征处理
        data_stamp = self.time_features(df_stamp['date'], freq='h')

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def time_features(self, dates, freq='h'):
        # 简单实现：Month, Day, Weekday, Hour
        dates = pd.to_datetime(dates.values)
        return np.vstack([
            dates.month,
            dates.day,
            dates.weekday,
            dates.hour
        ]).transpose()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ==========================================
# 3. 基础模块 (Embeddings & Layers)
# ==========================================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        # 4 features: month, day, weekday, hour
        self.month_embed = nn.Embedding(13, d_model)
        self.day_embed = nn.Embedding(32, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        self.hour_embed = nn.Embedding(24, d_model)

    def forward(self, x):
        x = x.long()
        # x shape: [Batch, Seq_len, 4]
        return self.month_embed(x[:, :, 0]) + self.day_embed(x[:, :, 1]) + \
            self.weekday_embed(x[:, :, 2]) + self.hour_embed(x[:, :, 3])


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ==========================================
# 4. 核心 Attention 模块 (ProbSparse)
# ==========================================
class ProbSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QE(self, Q, K, sample_k, n_top):
        # Q: [B, H, L_q, D], K: [B, H, L_k, D]
        B, H, L_k, E = K.shape
        _, _, L_q, _ = Q.shape

        # 随机采样 Key
        K_expand = K.unsqueeze(-3).expand(B, H, L_q, L_k, E)
        index_sample = torch.randint(L_k,
                                     (L_q, sample_k))  # Real sample logic is more complex in official, simplified here
        K_sample = K_expand[:, :, torch.arange(L_q).unsqueeze(1), index_sample, :]

        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # 稀疏度量 M = max(Q*K) - mean(Q*K)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_k)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the reduced Q to calculate attention
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            return V.cumsum(dim=-2) / torch.arange(1, L_V + 1).type_as(V).view(L_V, 1)  # Mean value

        # For masked attention (decoder), logic is complex, returning mean for simplicity in barebones
        return V.mean(dim=-2, keepdim=True).expand(B, H, L_Q, D).clone()

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QE(queries, keys, U_part, u)

        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Apply context
        context = self._get_initial_context(values, L_Q)  # Initialize with mean

        # Update context with Top-u attention
        attn = torch.softmax(scores_top, dim=-1)  # [B, H, u, L_k]

        # Map back to original Q dimension
        context_update = torch.matmul(attn, values)  # [B, H, u, D]

        # Scatter update
        context.scatter_(2, index.unsqueeze(-1).expand(-1, -1, -1, D), context_update)

        return context.transpose(1, 2).contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# ==========================================
# 5. Encoder & Decoder Modules
# ==========================================
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self Attention
        x = x + self.dropout(self.self_attention(x, x, x, x_mask)[0])
        x = self.norm1(x)
        # Cross Attention
        x = x + self.dropout(self.cross_attention(x, cross, cross, cross_mask)[0])
        x = self.norm2(x)
        # Feed Forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ==========================================
# 6. Informer 模型主体 (The Model)
# ==========================================
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, device=torch.device('cuda')):
        super(Informer, self).__init__()
        self.pred_len = pred_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # Attention Choice
        Attn = ProbSparseAttention if attn == 'prob' else None  # Fallback logic needed if Full

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# ==========================================
# 7. 实验控制 (Experiment & Training)
# ==========================================
class Exp_Informer:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = Informer(
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.device
        )
        return model

    def _get_data(self, flag):
        args = self.args
        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;

        data_set = Dataset_ETTh1(
            root_path='.',
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features='M',  # Multivariate
            target=args.target
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print(f"Start Training... Total Epochs: {self.args.train_epochs}")
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input: 填充 Context + 0
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"\tEpoch: {epoch + 1}, Step: {i + 1}, Loss: {loss.item():.5f}")

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.5f} Vali Loss: {vali_loss:.5f}")

            # Save best model logic (simplified)
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')

        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        preds = []
        trues = []

        print("Start Testing...")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Inverse Scaling if needed (Usually metrics are calculated on scaled data in papers,
                # but for real application you inverse transform first)
                # Here we calculate on SCALED data as per standard benchmarks

                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # Metrics
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        print(f'Test Results >> MSE: {mse:.4f}, MAE: {mae:.4f}')
        return mse, mae


# ==========================================
# 8. Main Execution
# ==========================================
if __name__ == '__main__':
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found. Please place the ETTh1.csv file in the current directory.")
    else:
        exp = Exp_Informer(args)

        print(">>>>>>> Start Training >>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train()

        print(">>>>>>> Start Testing >>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.test()