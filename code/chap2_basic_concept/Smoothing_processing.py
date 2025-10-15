import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TimeSeriesSmoothing:
    def __init__(self, data):
        """
        初始化时间序列平滑类
        :param data: 传入的时间序列数据，应该是pandas的Series或DataFrame的一列
        """
        self.data = data

    # 简单移动平均 (Simple Moving Average)
    def simple_moving_average(self, window):
        """
        简单移动平均方法
        :param window: 移动窗口的大小
        :return: 平滑后的时间序列
        """
        return self.data.rolling(window=window).mean()

    # 加权移动平均 (Weighted Moving Average)
    def weighted_moving_average(self, window, weights=None):
        """
        加权移动平均方法
        :param window: 移动窗口的大小
        :param weights: 权重值，默认为均等权重
        :return: 平滑后的时间序列
        """
        if weights is None:
            weights = np.ones(window)  # 默认的权重是均等的
        else:
            weights = np.array(weights)  # 将权重转换为numpy数组
        return self.data.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # 一次指数平滑 (Single Exponential Smoothing)
    def exponential_smoothing(self, alpha):
        """
        一次指数平滑方法
        :param alpha: 平滑因子，取值范围[0,1]
        :return: 平滑后的时间序列
        """
        return self.data.ewm(alpha=alpha, adjust=False).mean()

    # 二次指数平滑 (Double Exponential Smoothing)
    def double_exponential_smoothing(self, alpha, beta):
        """
        二次指数平滑方法
        :param alpha: 平滑因子
        :param beta: 趋势平滑因子
        :return: 平滑后的时间序列
        """
        # 计算水平成分（Level）
        level = self.data.ewm(alpha=alpha, adjust=False).mean()
        # 计算趋势成分（Trend）
        trend = level.diff().ewm(alpha=beta, adjust=False).mean()
        return level + trend

    # 三次指数平滑（Holt-Winters）
    def triple_exponential_smoothing(self, alpha, beta, gamma, season_length):
        """
        三次指数平滑（Holt-Winters）方法
        :param alpha: 平滑因子
        :param beta: 趋势平滑因子
        :param gamma: 季节性平滑因子
        :param season_length: 季节长度
        :return: 平滑后的时间序列
        """
        # 计算水平成分（Level）
        level = self.data.ewm(alpha=alpha, adjust=False).mean()
        # 计算趋势成分（Trend）
        trend = level.diff().ewm(alpha=beta, adjust=False).mean()
        # 计算季节性成分（Seasonality）
        seasonality = self.data - level - trend
        # 对季节性成分进行滚动平均来估计季节性模式
        seasonality_avg = seasonality.rolling(window=season_length).mean().shift(-season_length)
        return level + trend + seasonality_avg

    # 可视化所有平滑后的时间序列
    def plot_smoothed_series(self, smoothed_series_dict):
        """
        可视化平滑后的时间序列
        :param smoothed_series_dict: 包含每种平滑方法及其结果的字典
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data, label="Original Data", color='blue', linewidth=2)

        for label, smoothed_series in smoothed_series_dict.items():
            plt.plot(smoothed_series, label=label, linewidth=2)

        plt.title("Smoothed Time Series")
        plt.legend(loc="best")
        plt.show()

    # 保存平滑结果到文件
    def save_smoothed_series(self, smoothed_series_dict, save_dir='chap2_basic_concept/data/Smoothing_Results'):
        """
        保存平滑后的时间序列到文件
        :param smoothed_series_dict: 包含每种平滑方法及其结果的字典
        :param save_dir: 保存文件的目录
        """
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for label, smoothed_series in smoothed_series_dict.items():
            file_name = f"{save_dir}/{label.replace(' ', '_')}.csv"
            smoothed_series.reset_index().to_csv(file_name, index=False)

# 示例使用
if __name__ == "__main__":
    # 读取ETTm2.csv数据集，将date列解析为日期时间类型并设置为索引
    file_path = 'chap2_basic_concept/data/ETTm2.csv'
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # 选择一个字段作为时间序列数据，这里选择 'OT' 字段
    data = df['OT']
    print("原始数据:\n", data)

    # 创建TimeSeriesSmoothing对象
    smoothing = TimeSeriesSmoothing(data)

    # 使用不同的平滑方法
    smoothed_sma = smoothing.simple_moving_average(window=3)
    smoothed_wma = smoothing.weighted_moving_average(window=3, weights=[0.1, 0.3, 0.6])
    smoothed_es = smoothing.exponential_smoothing(alpha=0.3)
    smoothed_des = smoothing.double_exponential_smoothing(alpha=0.3, beta=0.3)
    smoothed_holt_winters = smoothing.triple_exponential_smoothing(alpha=0.3, beta=0.3, gamma=0.3, season_length=3)

    # 组织所有平滑结果到一个字典中
    smoothed_series_dict = {
        "Simple Moving Average": smoothed_sma,
        "Weighted Moving Average": smoothed_wma,
        "Exponential Smoothing": smoothed_es,
        "Double Exponential Smoothing": smoothed_des,
        "Holt-Winters (Triple Exponential Smoothing)": smoothed_holt_winters
    }

    # 保存平滑结果到文件
    smoothing.save_smoothed_series(smoothed_series_dict)

    # 可视化所有平滑结果
    smoothing.plot_smoothed_series(smoothed_series_dict)