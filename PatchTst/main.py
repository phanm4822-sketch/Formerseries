import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import time


# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    # 数据相关
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, 'data', 'ETTh1.csv')
    # data_path = 'ETTh1.csv'
    seq_len = 96  # 输入长度 (Lookback window)
    pred_len = 96  # 预测长度 (Horizon)

    # PatchTST 核心参数
    patch_len = 16  # 每个 Patch 的长度
    stride = 8  # Patch 之间的步长 (stride < patch_len 意味着重叠)

    # 模型架构参数
    enc_in = 7  # 输入变量数量 (ETTh1 有 7 个变量)
    d_model = 128  # 隐层维度 (Latent dimension)
    n_heads = 4  # Multi-head attention 的头数
    e_layers = 3  # Encoder 层数
    d_ff = 256  # FFN 中间层维度
    dropout = 0.2  # Dropout 比率
    head_dropout = 0.0  # 预测头的 Dropout

    # 训练参数
    batch_size = 32  # 批次大小 (如果显存不够可调小)
    epochs = 10  # 训练轮数 (演示用 10，实际可更多)
    learning_rate = 0.0001
    patience = 3  # 早停机制耐心值
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = Config()
print(f"Running on device: {args.device}")


# ==========================================
# 2. 数据加载与预处理 (Dataset & DataLoader)
# ==========================================
class Dataset_ETT_hour(Dataset):

    """
    标准的 ETT 数据集加载器。
    划分规则:
    - Train: 12个月
    - Val: 4个月
    - Test: 4个月
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path=Config.data_path,
                 target='OT', scale=True):
        # size: [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]  # PatchTST 通常不需要 label_len，这里保留兼容性
        self.pred_len = size[2]

        # type map
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
        df_raw = pd.read_csv(self.data_path)

        # 划分边界 (ETT-hour 的标准划分)
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 选取特征列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 去掉 date
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 训练集拟合 Scaler
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ==========================================
# 3. 模型组件 (Model Components)
# ==========================================

class RevIN(nn.Module):
    """
    可逆实例归一化 (Reversible Instance Normalization)
    用于解决时间序列中的分布偏移 (Distribution Shift) 问题。
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * 0)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x


class PatchEmbedding(nn.Module):
    """
    Patch Embedding 层
    功能: 将时间序列切片 (Patching) 并映射到隐层维度 (Projection)。
    """

    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching 这里的实现很巧妙：使用 Conv1d 来实现滑动窗口
        # Kernel size = patch_len, Stride = stride
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))  # 简单的 padding 策略

        # Projection: 将 patch_len 维度的向量映射到 d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional Embedding: 学习位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, d_model))  # 预设一个足够大的长度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch * n_vars, seq_len, 1]  <-- 注意这里已经是 Channel Independent 后的形状

        # 1. Patching
        # 我们手动做 Unfold 或者使用 View 操作。
        # 官方代码通常先把数据 unfold 成 (B, N, P)
        n_vars = x.shape[1]  # 这里其实是 1

        # 这里的输入 x 是 (Batch*M, Seq_Len, 1)
        # 调整维度为 (Batch*M, 1, Seq_Len) 以便进行 unfold
        x = x.permute(0, 2, 1)

        # Unfold: (Batch*M, 1, Seq_Len) -> (Batch*M, 1, Num_Patches, Patch_Len)
        # 计算 Patch 数量: num_patches = (seq_len - patch_len) / stride + 1
        # 简单起见，我们假设 seq_len 能被整除或处理好了 Padding
        # 这里使用 Unfold 算子
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x shape: (Batch*M, 1, Num_Patches, Patch_Len)

        x = x.permute(0, 2, 3, 1).squeeze(-1)
        # x shape: (Batch*M, Num_Patches, Patch_Len)

        # 2. Projection
        x = self.value_embedding(x)
        # x shape: (Batch*M, Num_Patches, d_model)

        # 3. Positional Embedding
        x = x + self.position_embedding[:, :x.shape[1], :]

        return self.dropout(x)


class FlattenHead(nn.Module):
    """
    Flatten Head
    功能: 将 Transformer 输出的所有 Patch 展平，并通过线性层直接预测未来所有时间步。
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super(FlattenHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [Batch * n_vars, Num_Patches, d_model]
        x = self.flatten(x)
        # x: [Batch * n_vars, Num_Patches * d_model]
        x = self.linear(x)
        # x: [Batch * n_vars, target_window]
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    PatchTST 完整模型
    """

    def __init__(self, config):
        super(PatchTST, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model

        # 1. RevIN
        self.revin = RevIN(config.enc_in)

        # 计算 Patch 数量
        self.num_patches = int((config.seq_len - config.patch_len) / config.stride + 1)

        # 2. Backbone (Transformer Encoder)
        # PatchTST 的核心：Embedding -> Encoder -> Head
        self.patch_embedding = PatchEmbedding(config.d_model, config.patch_len, config.stride, config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,
                                                   dim_feedforward=config.d_ff, dropout=config.dropout,
                                                   activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)

        # 3. Head
        self.head = FlattenHead(config.enc_in, self.num_patches * config.d_model, config.pred_len, config.head_dropout)

    def forward(self, x):
        # Input x: [Batch, Seq_Len, N_Vars]

        # Step 1: RevIN Normalization
        x = self.revin(x, 'norm')

        # Step 2: Channel Independence (关键步骤!)
        # 变换: (Batch, Seq_Len, N_Vars) -> (Batch * N_Vars, Seq_Len, 1)
        B, L, M = x.shape
        x = x.permute(0, 2, 1).reshape(B * M, L, 1)

        # Step 3: Patching & Embedding
        # Output: (Batch * N_Vars, Num_Patches, d_model)
        x_enc = self.patch_embedding(x)

        # Step 4: Transformer Encoder
        # Output: (Batch * N_Vars, Num_Patches, d_model)
        # 注意：这里不需要 Mask，因为我们看得到整个过去的历史
        x_enc = self.encoder(x_enc)

        # Step 5: Flatten Head & Prediction
        # Output: (Batch * N_Vars, Pred_Len)
        dec_out = self.head(x_enc)

        # Step 6: Reshape back (Channel Dependence recovery)
        # (Batch * N_Vars, Pred_Len) -> (Batch, N_Vars, Pred_Len) -> (Batch, Pred_Len, N_Vars)
        dec_out = dec_out.reshape(B, M, -1).permute(0, 2, 1)

        # Step 7: RevIN Denormalization
        dec_out = self.revin(dec_out, 'denorm')

        return dec_out


# ==========================================
# 4. 训练与评估辅助函数
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
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


# ==========================================
# 5. 主程序 (Main Execution)
# ==========================================

# 准备数据
print("Preparing data...")
train_dataset = Dataset_ETT_hour(root_path='./', flag='train', size=[args.seq_len, 0, args.pred_len],
                                 data_path=args.data_path)
val_dataset = Dataset_ETT_hour(root_path='./', flag='val', size=[args.seq_len, 0, args.pred_len],
                               data_path=args.data_path)
test_dataset = Dataset_ETT_hour(root_path='./', flag='test', size=[args.seq_len, 0, args.pred_len],
                                data_path=args.data_path)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)  # Batch size 1 for visualization

# 初始化模型
model = PatchTST(args).to(args.device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
early_stopping = EarlyStopping(patience=args.patience, verbose=True)

# 训练循环
print("Start training...")
train_steps = len(train_loader)

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

for epoch in range(args.epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        iter_count += 1
        optimizer.zero_grad()

        batch_x = batch_x.float().to(args.device)
        batch_y = batch_y.float().to(args.device)

        # Forward
        outputs = model(batch_x)

        # Loss calculation (f_dim='M')
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

    train_loss = np.average(train_loss)

    # Validation
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_loader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss.append(loss.item())
    val_loss = np.average(val_loss)

    print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {val_loss:.7f}")

    # Early Stopping check
    early_stopping(val_loss, model, './checkpoints')
    if early_stopping.early_stop:
        print("Early stopping")
        break

# 载入最优模型进行测试
print("Loading best model for testing...")
model.load_state_dict(torch.load('./checkpoints/checkpoint.pth'))
model.eval()

preds = []
trues = []

print("Start testing...")
with torch.no_grad():
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(args.device)
        batch_y = batch_y.float().to(args.device)

        outputs = model(batch_x)

        pred = outputs.detach().cpu().numpy()
        true = batch_y.detach().cpu().numpy()

        preds.append(pred)
        trues.append(true)

# 结果处理
preds = np.array(preds)
trues = np.array(trues)
print(f'Test shape: {preds.shape}')  # (N, 1, 96, 7)
preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
print(f'Test shape reshaped: {preds.shape}')

# 计算 Metrics
mae = np.mean(np.abs(preds - trues))
mse = np.mean((preds - trues) ** 2)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

# 可视化: 随机选一个样本的 OT 列 (最后一列) 进行展示
idx = 0
plt.figure(figsize=(12, 6))
plt.plot(trues[idx, :, -1], label='GroundTruth')
plt.plot(preds[idx, :, -1], label='Prediction')
plt.legend()
plt.title(f'PatchTST Prediction (Sample {idx}, Variable OT)')
plt.savefig('patchtst_prediction.png')
print("Visualization saved to patchtst_prediction.png")