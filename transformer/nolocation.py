import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import os


# ==========================================
# 配置参数
# ==========================================
class Config:
    seq_len = 96  # 输入序列长度
    pred_len = 96  # 预测序列长度 (已修改为 96)
    label_len = 48  # Decoder输入的已知序列长度

    d_model = 64
    n_head = 4
    e_layers = 1
    d_layers = 1
    d_ff = 256
    dropout = 0.05

    batch_size = 32
    epochs = 2
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 1. 数据加载 (保持不变)
# ==========================================
class ETTH1Dataset(Dataset):
    def __init__(self, data_path, flag='train', seq_len=96, pred_len=96):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 这里的路径如果不确定，请使用绝对路径或相对路径 'data/ETTh1.csv'
        try:
            self.df_raw = pd.read_csv(data_path)
        except FileNotFoundError:
            # 兼容可能的路径问题， fallback到当前目录
            self.df_raw = pd.read_csv('ETTh1.csv')

        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        cols_data = self.df_raw.columns[1:]
        self.data_data = self.df_raw[cols_data].values

        self.scaler = StandardScaler()
        train_data = self.data_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data_x = self.scaler.transform(self.data_data[self.border1:self.border2])
        self.data_y = self.data_x

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


# ==========================================
# 2. 模型核心组件 (PositionalEncoding类虽然保留，但在Transformer中不再调用)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.tokenConv(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, _ = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, _ = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, _ = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


# ==========================================
# 3. 完整的 Transformer 模型 (修改版)
# ==========================================
class Transformer(nn.Module):
    def __init__(self, c_in, c_out, seq_len, pred_len, d_model, n_head, e_layers, d_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.pred_len = pred_len

        # Encoding Part
        self.enc_embedding = TokenEmbedding(c_in, d_model)

        # 【修改点 1】: 注释掉 Positional Encoding 的初始化
        # self.pos_embedding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, n_head, d_model // n_head, d_model // n_head, dropout)
            for _ in range(e_layers)
        ])

        # Decoding Part
        self.dec_embedding = TokenEmbedding(c_in, d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, n_head, d_model // n_head, d_model // n_head, dropout)
            for _ in range(d_layers)
        ])

        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_dec):
        # 1. Encoder
        enc_out = self.enc_embedding(x_enc)

        # 【修改点 2】: 移除位置编码的加法操作
        # enc_out = self.pos_embedding(enc_out)

        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        # 2. Decoder
        dec_out = self.dec_embedding(x_dec)

        # 【修改点 3】: 移除位置编码的加法操作
        # dec_out = self.pos_embedding(dec_out)

        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)

        output = self.projection(dec_out)
        return output[:, -self.pred_len:, :]


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    args = Config()

    # 请确保 'data/ETTh1.csv' 路径正确，如果报错请改为相对路径 'ETTh1.csv'
    train_set = ETTH1Dataset('data/ETTh1.csv', flag='train', seq_len=args.seq_len, pred_len=args.pred_len)
    test_set = ETTH1Dataset('data/ETTh1.csv', flag='test', seq_len=args.seq_len, pred_len=args.pred_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = Transformer(
        c_in=7, c_out=7,
        seq_len=args.seq_len, pred_len=args.pred_len,
        d_model=args.d_model, n_head=args.n_head,
        e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, dropout=args.dropout
    ).to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Start Training (pred_len={args.pred_len}, NO Positional Encoding)...")
    model.train()
    for epoch in range(args.epochs):
        train_loss = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)

            # Decoder Input
            dec_input_token = batch_x[:, -args.label_len:, :]
            dec_zeros = torch.zeros(batch_x.size(0), args.pred_len, batch_x.size(2)).to(args.device)
            dec_input = torch.cat([dec_input_token, dec_zeros], dim=1)

            optimizer.zero_grad()
            outputs = model(batch_x, dec_input)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item():.4f}")

    print("\nEvaluating...")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            dec_input_token = batch_x[:, -args.label_len:, :]
            dec_zeros = torch.zeros(batch_x.size(0), args.pred_len, batch_x.size(2)).to(args.device)
            dec_input = torch.cat([dec_input_token, dec_zeros], dim=1)
            outputs = model(batch_x, dec_input)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))

    print(f"Final Results (No PE) -> MSE: {mse:.4f}, MAE: {mae:.4f}")