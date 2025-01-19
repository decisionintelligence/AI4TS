import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# 1. 基于传统统计的平稳化方法

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
    '''
    return np.log(series).dropna()

# 标准归一化（Z-score标准化）
def z_score(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：z-score方法
    '''
    scaler = StandardScaler()
    return pd.Series(scaler.fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)

# 2. 基于深度学习的平稳化方法

# Deep Adaptive Input Normalization (DAIN)
def DAIN(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：DAIN (Deep Adaptive Input Normalization)
    
    DAIN方法通过神经网络自动学习输入数据的适应性标准化。实际实现需要使用合适的深度神经网络架构，如卷积神经网络或循环神经网络，
    对时序数据进行学习和规范化。模型的训练过程将涉及对时间序列的拟合与调整。
    '''
    # 伪代码：这是一个简单的模拟实现，实际方法会依赖深度学习架构和训练过程
    model = nn.Sequential(
        nn.Linear(1, 64),  # 输入层，接受单一时序数据
        nn.ReLU(),         # 激活函数，非线性变换
        nn.Linear(64, 1)   # 输出层，输出处理后的时序数据
    )
    # 将时序数据转换为张量
    series_tensor = torch.tensor(series.values.reshape(-1, 1), dtype=torch.float32)
    return pd.Series(model(series_tensor).detach().numpy().flatten(), index=series.index)

# Reversible Instance Normalization (RevIN)
def RevIN(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：RevIN (Reversible Instance Normalization)
    
    RevIN通过对每个时序数据实例进行归一化和反归一化处理，调整输入数据的分布。此方法主要通过卷积网络或类似的架构来实现。
    需要训练过程来学习输入数据的反归一化映射。
    '''
    # 伪代码：这里只是一个简单模型的模拟，实际的RevIN方法使用更复杂的网络结构和训练流程
    model = nn.Sequential(
        nn.LayerNorm(1),   # 层归一化
        nn.ReLU(),         # 激活函数
        nn.Linear(1, 1)    # 输出层
    )
    series_tensor = torch.tensor(series.values.reshape(-1, 1), dtype=torch.float32)
    return pd.Series(model(series_tensor).detach().numpy().flatten(), index=series.index)

# Non-Stationary Transformer
def Non_Stationary_Transformer(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：Non-Stationary Transformer
    
    使用Transformer模型，特别是改进版的非平稳时序变换器，通过自注意力机制来处理时间序列数据的长距离依赖。实际实现中，
    需要训练Transformer模型，并通过其编码器对时序数据进行变换。
    '''
    # 伪代码：这是一个简化的模型，真实的Non-Stationary Transformer方法会依赖训练好的Transformer架构
    model = nn.Sequential(
        nn.Linear(1, 64),  # 输入层
        nn.ReLU(),         # 激活函数
        nn.Linear(64, 1)   # 输出层
    )
    series_tensor = torch.tensor(series.values.reshape(-1, 1), dtype=torch.float32)
    return pd.Series(model(series_tensor).detach().numpy().flatten(), index=series.index)

# Shard-wise Adaptive Normalization (SAN)
def SAN(series):
    '''
    输入：非平稳时序
    输出：处理后的平稳时序
    方法：SAN (Shard-wise Adaptive Normalization)
    
    SAN方法通过将时序数据分成多个子序列（分片），并对每个分片进行独立的适应性归一化，来处理复杂的时序数据。实际的实现过程包括数据的分片与归一化操作。
    该方法通常需要训练一个深度神经网络来学习这些分片的特征。
    '''
    # 伪代码：模拟实现，实际方法会依赖深度学习模型进行训练
    model = nn.Sequential(
        nn.Linear(1, 128),  # 输入层
        nn.ReLU(),          # 激活函数
        nn.Linear(128, 1)   # 输出层
    )
    series_tensor = torch.tensor(series.values.reshape(-1, 1), dtype=torch.float32)
    return pd.Series(model(series_tensor).detach().numpy().flatten(), index=series.index)

# 示例用法
if __name__ == "__main__":
    # 生成一个示例时序数据
    data = pd.Series([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])

    print("原始数据:\n", data)

    # 差分法
    print("差分法:\n", diff(data))

    # 对数转换法
    print("对数转换法:\n", log_transform(data))

    # Z-score标准化
    print("Z-score标准化:\n", z_score(data))

    # # DAIN
    # print("DAIN:\n", DAIN(data))

    # # RevIN
    # print("RevIN:\n", RevIN(data))

    # # Non-Stationary Transformer
    # print("Non-Stationary Transformer:\n", Non_Stationary_Transformer(data))

    # # SAN
    # print("SAN:\n", SAN(data))
