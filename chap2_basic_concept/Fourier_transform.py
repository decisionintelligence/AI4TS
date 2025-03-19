import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def fourier_transform(series):
    """
    对时域数据进行傅里叶变换，转换为频域数据。

    参数:
        series: pd.Series, 输入的时间序列数据，索引需为时间格式。
    
    返回:
        freq: np.ndarray, 非负频率数组 (单位: Hz)
        amplitude: np.ndarray, 对应频率下的幅值
    """
    if not isinstance(series, pd.Series):
        raise ValueError("输入数据必须是 pandas Series 类型")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series 索引必须为 DatetimeIndex")

    # 计算采样时间间隔（假定时间均匀采样），单位：秒
    dt = (series.index[1] - series.index[0]).total_seconds()
    
    # 进行傅里叶变换
    fft_result = np.fft.fft(series.values)
    fft_freq = np.fft.fftfreq(len(series), d=dt)

    # 取非负频率部分
    mask = fft_freq >= 0
    freq = fft_freq[mask]
    amplitude = np.abs(fft_result[mask])

    return freq, amplitude


def save_fourier_results(freq, amplitude, save_dir, filename="fourier_transform.csv"):
    """
    保存傅里叶变换结果为 CSV 文件。

    参数:
        freq: np.ndarray, 频率数组
        amplitude: np.ndarray, 幅值数组
        save_dir: str, 结果保存的目录路径
        filename: str, 结果文件名，默认为 "fourier_transform.csv"
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fourier_df = pd.DataFrame({'Frequency (Hz)': freq, 'Amplitude': amplitude})
    save_path = os.path.join(save_dir, filename)
    fourier_df.to_csv(save_path, index=False)
    print(f"傅里叶变换结果已保存至: {save_path}")



if __name__ == "__main__":
    # 示例：读取数据并执行傅里叶变换
    file_path = "chap2_basic_concept/data/ETTm2.csv"
    save_dir = "chap2_basic_concept/data/Fourier_Results"

    # 读取 CSV 文件
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # 选择 "OT" 列进行傅里叶变换
    column_name = "OT"
    data_series = df[column_name]

    # 计算傅里叶变换
    freq, amplitude = fourier_transform(data_series)

    # 保存结果
    save_fourier_results(freq, amplitude, save_dir)


