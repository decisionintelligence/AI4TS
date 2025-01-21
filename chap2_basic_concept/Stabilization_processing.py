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

