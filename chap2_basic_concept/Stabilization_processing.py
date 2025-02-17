import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

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
    # 对数据进行标准化处理
    standardized_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    # 将标准化后的数据转换为pandas的Series对象，并指定列名
    return pd.Series(standardized_data, index=series.index, name='OT_Standardized')


# 示例用法
if __name__ == "__main__":
    # 读取ETTm2.csv数据集，将date列解析为日期时间类型并设置为索引
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

    # Z-score标准化
    z_score_result = z_score(data).reset_index()
    z_score_result.to_csv(os.path.join(save_dir, 'z_score_result.csv'), index=False)