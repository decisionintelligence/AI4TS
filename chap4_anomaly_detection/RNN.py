## 导入相关库
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn.functional as F
import copy

## 定义数据集(Dataset)类
class ReconstructionDataset(Dataset):
    def __init__(self, data_path, seq_len, split='train'):
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.seq_len = seq_len

        df_raw = pd.read_csv(data_path)

        border1s = [0, 12*30*24*4 - seq_len, 12*30*24*4 + 4*30*24*4 - seq_len]
        border2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        self.scaler = StandardScaler()
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)

        data = self.scaler.transform(df_data.values)
        self.data = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data[s_begin:s_end]
        return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_x).float()

    def __len__(self):
        return len(self.data) - self.seq_len + 1


## 定义 RNN 自编码器模型（双向编码 + 解码）
class ReconstructionRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super(ReconstructionRNN, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # 编码阶段
        enc_out, (h_n, c_n) = self.encoder(x)

        # 解码阶段，使用编码后的隐状态作为解码初始输入
        dec_input = enc_out  # 这里直接用编码后的序列作为解码输入
        dec_out, _ = self.decoder(dec_input, (h_n, c_n))

        # 输出映射回原始维度
        out = self.output_layer(dec_out)
        return out


## 超参配置
epochs = 10
seq_len = 96
batch_size = 64
learning_rate = 1e-3
data_path = r'D:\OverLeaf写书\Time_Series_Analytics-main\Time_Series_Analytics-main\chap4_anomaly_detection\data\ETTm2.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## 加载数据
train_dataset = ReconstructionDataset(data_path, seq_len, 'train')
val_dataset   = ReconstructionDataset(data_path, seq_len, 'val')
test_dataset  = ReconstructionDataset(data_path, seq_len, 'test')

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader   = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

## 定义模型、损失、优化器
input_dim = train_dataset.data.shape[1]
model = ReconstructionRNN(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## 训练
best_val_loss = float('inf')
best_model_state = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"--> Best model found at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

## 测试 + 异常检测
model.load_state_dict(best_model_state)
model.eval()

reconstruction_errors = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        batch_errors = F.mse_loss(outputs, batch_y, reduction='none')
        batch_errors = batch_errors.mean(dim=[1,2])
        reconstruction_errors.extend(batch_errors.cpu().numpy())

# 阈值 = 均值 + 3σ
threshold = np.mean(reconstruction_errors) + 3*np.std(reconstruction_errors)

# 标记异常
anomaly_labels = (np.array(reconstruction_errors) > threshold).astype(int)
print(f"检测到异常数量: {anomaly_labels.sum()} / {len(anomaly_labels)}")
