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

## 定义数据集(Dataset)
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


## 定义 VAE 模型
class ReconstructionVAE(nn.Module):
    def __init__(self, seq_len, input_dim, latent_dim=32, hidden_dim=128):
        super(ReconstructionVAE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.input_size = seq_len * input_dim

        # 编码器
        self.fc1 = nn.Linear(self.input_size, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.input_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch, seq_len * input_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        recon_x = recon_x.view(batch_size, self.seq_len, self.input_dim)
        return recon_x, mu, logvar


## 超参配置
epochs = 10
seq_len = 96
batch_size = 64
learning_rate = 1e-3
data_path = 'chap4_anomaly_detection\data\ETTm2.csv'
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
model = ReconstructionVAE(seq_len, input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## 自定义 VAE 损失 (重构误差 + KL 散度)
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss, recon_loss.item()

## 训练
best_val_loss = float('inf')
best_model_state = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        recon_x, mu, logvar = model(batch_x)
        loss, recon_loss_value = vae_loss(recon_x, batch_x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(recon_loss_value)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            recon_x, mu, logvar = model(batch_x)
            loss, recon_loss_value = vae_loss(recon_x, batch_x, mu, logvar)
            val_losses.append(recon_loss_value)

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}/{epochs}, Train Recon Loss: {avg_train_loss:.4f}, Val Recon Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"--> Best model found at epoch {epoch+1} with Val Recon Loss: {best_val_loss:.4f}")

## 测试 + 异常检测
model.load_state_dict(best_model_state)
model.eval()

reconstruction_errors = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        recon_x, mu, logvar = model(batch_x)
        batch_errors = F.mse_loss(recon_x, batch_x, reduction='none')
        batch_errors = batch_errors.mean(dim=[1,2])
        reconstruction_errors.extend(batch_errors.cpu().numpy())

# 阈值 = 均值 + 3σ
threshold = np.mean(reconstruction_errors) + 3*np.std(reconstruction_errors)

# 标记异常
anomaly_labels = (np.array(reconstruction_errors) > threshold).astype(int)
print(f"检测到异常数量: {anomaly_labels.sum()} / {len(anomaly_labels)}")
