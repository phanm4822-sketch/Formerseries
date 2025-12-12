import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import time
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ========================================================
# 1. 配置参数 (Configuration)
#    模拟 argparse，方便在 Notebook 中直接修改
# ========================================================
class Config:
    def __init__(self):
        # 数据相关
        self.root_path = './'
        self.data_path = 'ETTh1.csv'
        self.task_name = 'long_term_forecast'
        self.model_id = 'ETTh1_96_96'
        self.model = 'iTransformer'
        self.data = 'ETTh1'
        self.features = 'M'  # M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量
        self.target = 'OT'  # 目标列
        self.freq = 'h'  # 时间频率
        self.checkpoints = './checkpoints/'

        # 预测任务相关
        self.seq_len = 96  # 输入长度
        self.label_len = 48  # 用于作为Decoder输入的已知长度(iTransformer其实主要用seq_len)
        self.pred_len = 96  # 预测长度

        # 模型超参数
        self.enc_in = 7  # Encoder输入特征数 (ETTh1有7列)
        self.dec_in = 7  # Decoder输入特征数
        self.c_out = 7  # 输出特征数
        self.d_model = 512  # 模型维度
        self.n_heads = 8  # 多头注意力的头数
        self.e_layers = 2  # Encoder层数
        self.d_ff = 2048  # FFN维度
        self.dropout = 0.1
        self.use_norm = True  # 是否使用归一化 (RevIN核心)

        # 优化相关
        self.num_workers = 0  # Windows下建议设为0
        self.itr = 1  # 实验运行次数
        self.train_epochs = 10
        self.batch_size = 32
        self.patience = 3  # Early Stopping
        self.learning_rate = 0.0001
        self.des = 'Exp'
        self.lradj = 'type1'
        self.use_amp = False  # 混合精度训练

        # GPU
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.device = torch.device('cuda:{}'.format(self.gpu)) if self.use_gpu else torch.device('cpu')


args = Config()


# ========================================================
# 2. 工具模块 (Utils)
#    包含时间特征提取、标准化、EarlyStopping
# ========================================================

class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def time_features(dates, freq='h'):
    # 生成时间特征：月、日、周、时等
    # 这里是一个简化版实现，对应官方 utils.timefeatures
    dates = pd.to_datetime(dates.values)
    return np.vstack([
        dates.month,
        dates.day,
        dates.weekday,
        dates.hour
    ]).transpose()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


# ========================================================
# 3. 数据加载模块 (Data Loader)
#    针对 ETTh1 的 Dataset 定义
# ========================================================

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size: [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # type_map 用于划分数据集
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ETTh1 的标准划分边界
        # 12个月训练, 4个月验证, 4个月测试
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

        # 时间特征编码
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

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


# ========================================================
# 4. 模型层定义 (Model Layers)
#    iTransformer 的组件
# ========================================================

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # iTransformer的关键：将整个时间序列长度 seq_len 映射为 d_model
        # 这里的 c_in 其实是 seq_len
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [Batch, Variates, Seq_Len]
        x = self.value_embedding(x)
        # x: [Batch, Variates, d_model]
        return self.dropout(x)


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

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 投影并分头
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = True
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
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

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for i, layer in enumerate(self.attn_layers):
            x, attn = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# ========================================================
# 5. iTransformer 模型主类
# ========================================================

class iTransformer(nn.Module):
    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.use_norm = configs.use_norm  # RevIN flag

        # Encoding: 映射 seq_len -> d_model
        # 注意：iTransformer 中，输入被转置，维度是 seq_len 被当做特征嵌入
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation='gelu'
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder / Projector: 映射 d_model -> pred_len
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization (RevIN part 1)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Inversion: [Batch, Seq_Len, Variates] -> [Batch, Variates, Seq_Len]
        # 这里的 B 是 Batch, L 是 Seq_Len, N 是 Variates
        _, _, N = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)

        # Embedding: [Batch, Variates, Seq_Len] -> [Batch, Variates, d_model]
        enc_out = self.enc_embedding(x_enc, None)  # 也可以传入 x_mark_enc 但核心是 inverted

        # Encoder: [Batch, Variates, d_model] -> [Batch, Variates, d_model]
        # Attention 发生在 Variates 之间
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection: [Batch, Variates, d_model] -> [Batch, Variates, Pred_Len]
        dec_out = self.projector(enc_out)

        # Restore Dimension: [Batch, Variates, Pred_Len] -> [Batch, Pred_Len, Variates]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization (RevIN part 2)
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None


# ========================================================
# 6. 实验流程 (Experiment)
#    管理数据加载、模型构建、训练、测试
# ========================================================

class Exp_Main():
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = iTransformer(self.args)
        return model

    def _get_data(self, flag):
        data_set = Dataset_ETT_hour(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            timeenc=0,  # 简单起见设为0
            freq=self.args.freq
        )
        print(f'Data flag: {flag}, len: {len(data_set)}')

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = 1  # 测试时通常设为1，或者 batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input (通常全0或部分历史+0)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 学习率调整 (简易版)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.array(preds)
        trues = np.array(trues)

        # 处理可能的 batch 维度不一致 (如果最后一个batch不满)
        # 这里的 reshape 需要小心，建议先 concat
        # 简单处理：
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse:.4f}, mae:{mae:.4f}')

        # 可视化第一条
        plt.figure()
        plt.plot(trues[0, :, -1], label='GroundTruth')
        plt.plot(preds[0, :, -1], label='Prediction')
        plt.legend()
        plt.show()

        return mse, mae


def adjust_learning_rate(optimizer, epoch, args):
    # type1
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


def metric(pred, true):
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true))
    mspe = np.mean(np.square((pred - true) / true))
    return mae, mse, rmse, mape, mspe


# ========================================================
# 7. 主执行入口 (Main)
# ========================================================

if __name__ == '__main__':
    # 设置随机种子，保证结果可复现
    fix_seed = 2024
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    print("Args in experiment:")
    print(args.__dict__)

    exp = Exp_Main(args)

    setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        1,  # factor (not used in full attention but keep for naming)
        args.dropout,
        'distil',  # not used
        args.des, 0)

    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)

    print(f'>>>>>>>testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.test(setting)