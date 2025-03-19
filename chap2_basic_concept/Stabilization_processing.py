import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 新增：引入PyTorch，用于实现DAIN模块
import torch
import torch.nn as nn

# -------------------------------
# 1. 基于传统统计的平稳化方法
# -------------------------------

# 差分法
def diff(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：差分法
    '''
    return series.diff().dropna()

# 对数转换法
def log_transform(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：对数转换法
    如果序列中存在非正值，则计算一个合适的偏移量，使得所有值均为正
    '''
    min_val = series.min()
    # 如果存在非正值，则偏移量为 abs(min_val) + epsilon
    if min_val <= 0:
        shift_value = abs(min_val) + 1e-5
    else:
        shift_value = 0
    return np.log(series + shift_value).dropna()


# 标准归一化（Z-score标准化）
def z_score(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：Z-score方法
    '''
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(standardized_data, index=series.index, name='OT_Standardized')

# -------------------------------
# 2. 深度自适应输入标准化 (DAIN)
# -------------------------------

class DAIN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        """
        构造函数：
        input_dim: 输入特征维度（对单变量时为1）
        hidden_dim: 隐藏层大小
        """
        super(DAIN, self).__init__()
        # 两个全连接网络分别生成动态缩放和平移参数
        self.fc_gamma = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        """
        输入：x，形状为 (batch_size, seq_len, input_dim)
        过程：
            1. 计算沿时间维度的均值 mu 和标准差 sigma
            2. 对x进行归一化： x_norm = (x - mu) / sigma
            3. 基于归一化后的全局统计信息生成动态的 gamma 和 beta
            4. 应用门控机制： x_adapt = gamma * x_norm + beta
        输出：归一化后的结果 x_adapt，形状与 x 相同
        """
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-5  # 防止除零
        x_norm = (x - mu) / sigma

        # 提取全局特征（这里简单地取时间维度的平均）
        global_feature = x_norm.mean(dim=1)  # shape: (batch_size, input_dim)

        # 生成动态参数
        gamma = self.fc_gamma(global_feature).unsqueeze(1)  # shape: (batch_size, 1, input_dim)
        beta = self.fc_beta(global_feature).unsqueeze(1)    # shape: (batch_size, 1, input_dim)

        x_adapt = gamma * x_norm + beta
        return x_adapt

def dain_transform(series, model=None):
    '''
    输入：
        series: 非平稳时序 pd.Series
        model: 可选的 DAIN 模型（若未提供，则初始化一个新的 DAIN 模型，注意模型参数未经过训练，仅作为示例）
    输出：
        处理后的平稳时序，返回类型为 pd.Series
    方法：
        利用 DAIN 模块对序列进行深度自适应输入标准化
    '''
    # 将序列转换为 PyTorch 张量，形状为 (1, seq_len, 1)
    x = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    if model is None:
        model = DAIN(input_dim=1, hidden_dim=32)
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        x_transformed = model(x)
    # 转换回 numpy 数组，再构造 pd.Series
    result = x_transformed.squeeze().numpy()
    return pd.Series(result, index=series.index, name='OT_DAIN')

# -------------------------------
# 3. 可逆实例规范化 (RevIN)
# -------------------------------

def revin_transform(series, gamma=1.0, beta=0.0):
    '''
    输入：
        series: 非平稳时序 pd.Series
        gamma: 自适应缩放参数（默认1.0，可根据需要调整或训练得到）
        beta: 自适应平移参数（默认0.0，可根据需要调整或训练得到）
    输出：
        adapted_series: 经过 RevIN 处理后的平稳时序（pd.Series）
        mu: 原始序列的均值
        sigma: 原始序列的标准差（加了微小常数防止除零）
    方法：
        1. 计算输入序列的均值 mu 和标准差 sigma
        2. 标准化： normalized = (series - mu) / sigma
        3. 引入自适应参数： adapted = gamma * normalized + beta
    '''
    mu = series.mean()
    sigma = series.std() + 1e-5
    normalized = (series - mu) / sigma
    adapted = gamma * normalized + beta
    adapted_series = pd.Series(adapted, index=series.index, name='OT_RevIN')
    return adapted_series, mu, sigma

def revin_inverse(normalized_series, mu, sigma, gamma=1.0, beta=0.0):
    '''
    输入：
        normalized_series: 经 RevIN 处理后的序列（pd.Series）
        mu: 原始序列均值
        sigma: 原始序列标准差
        gamma, beta: 与前向变换中相同的自适应参数
    输出：
        recovered_series: 逆变换恢复后的原始序列（pd.Series）
    方法：
        根据 RevIN 的前向变换公式逆向计算得到原始数据：
        original = sigma * ((normalized_series - beta) / gamma) + mu
    '''
    recovered = sigma * ((normalized_series - beta) / gamma) + mu
    recovered_series = pd.Series(recovered, index=normalized_series.index, name='OT_RevIN_Inverse')
    return recovered_series

def san_transform(series, num_shards=10):
    """
    分片归一化（SAN，Shard-wise Adaptive Normalization）
    
    参数：
        series: 待归一化的时间序列 (pd.Series)
        num_shards: 分片数量，默认划分为 10 个分片（如果序列长度不足，最后一个分片可能会包含余数数据）
    
    输出：
        normalized_series: 分片归一化后的时间序列 (pd.Series)，归一化方法为对每个分片使用 z-score 标准化
    
    方法：
        1. 将序列等分为 num_shards 个分片（最后一个分片包含剩余数据）。
        2. 对每个分片分别计算均值和标准差（防止除零时加上微小常数）。
        3. 使用公式 (x - mean) / (std + epsilon) 对分片数据进行归一化。
    """
    n = len(series)
    shard_length = n // num_shards
    normalized_data = []
    
    for i in range(num_shards):
        start_idx = i * shard_length
        # 最后一个分片取到序列末尾
        if i == num_shards - 1:
            shard = series.iloc[start_idx:]
        else:
            shard = series.iloc[start_idx:start_idx + shard_length]
        
        mean_val = shard.mean()
        std_val = shard.std() + 1e-5  # 加入微小常数避免除零
        normalized_shard = (shard - mean_val) / std_val
        normalized_data.append(normalized_shard)
    
    normalized_series = pd.concat(normalized_data)
    normalized_series.name = 'OT_SAN'
    return normalized_series



# -------------------------------
# 示例用法
# -------------------------------
if __name__ == "__main__":
    # 读取 ETTm2.csv 数据集，将 date 列解析为日期时间类型并设置为索引
    file_path = 'chap2_basic_concept/data/ETTm2.csv'
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # 选择一个字段作为时间序列数据，这里选择 'OT' 字段
    data = df['OT']

    # 确保保存结果的目录存在
    save_dir = 'chap2_basic_concept/data/Stationarization_Results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存原始数据
    original_data = data.reset_index()
    original_data.to_csv(os.path.join(save_dir, 'original_data.csv'), index=False)

    # 差分法
    diff_result = diff(data).reset_index()
    diff_result.to_csv(os.path.join(save_dir, 'diff_result.csv'), index=False)

    # 对数转换法
    log_result = log_transform(data).reset_index()
    log_result.to_csv(os.path.join(save_dir, 'log_result.csv'), index=False)

    # Z-score 标准化
    z_score_result = z_score(data).reset_index()
    z_score_result.to_csv(os.path.join(save_dir, 'z_score_result.csv'), index=False)

    # DAIN 标准化
    dain_result = dain_transform(data).reset_index()
    dain_result.to_csv(os.path.join(save_dir, 'dain_result.csv'), index=False)

    # RevIN 标准化
    revin_result, mu, sigma = revin_transform(data)
    revin_result_reset = revin_result.reset_index()
    revin_result_reset.to_csv(os.path.join(save_dir, 'revin_result.csv'), index=False)

    # 如有需要，进行 RevIN 的逆变换以恢复原始数据
    # recovered = revin_inverse(revin_result, mu, sigma)
    # recovered_reset = recovered.reset_index()
    # recovered_reset.to_csv(os.path.join(save_dir, 'revin_inverse_result.csv'), index=False)

    # 分片归一化（SAN）
    san_result = san_transform(data, num_shards=10).reset_index()
    san_result.to_csv(os.path.join(save_dir, 'san_result.csv'), index=False)

