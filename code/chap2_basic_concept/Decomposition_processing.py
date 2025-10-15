import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD

class TimeSeriesDecomposition:
    def __init__(self, data):
        """
        初始化时间序列分解对象
        :param data: pandas Series 或 DataFrame 的一列，代表时间序列数据
        """
        self.data = data  # 传入数据，假设数据为pandas的Series或DataFrame的一列

    # 经典分解法：加性模型
    def classical_decomposition(self, seasonal_periods=12):
        """
        经典分解法：加性模型进行时间序列分解
        :param seasonal_periods: 季节性周期，默认为12，通常对应年周期的月数据
        :return: trend（趋势项）, seasonal（季节项）, residual（残差项）
        """
        # 使用滚动平均法计算趋势
        trend = self.data.rolling(window=seasonal_periods).mean()

        # 季节项是原数据减去趋势项
        seasonal = self.data - trend

        # 残差项为原数据减去趋势项和季节项
        residual = self.data - trend - seasonal

        return trend, seasonal, residual

    # 基函数扩展（简化版，展示如何进行分解）
    def basis_function_expansion(self, num_components=3):
        """
        基函数扩展：使用多项式拟合进行分解
        :param num_components: 基函数的数量，默认为3（即三次多项式）
        :return: trend（趋势项）, residual（残差项）
        """
        # 使用sklearn的PolynomialFeatures进行多项式特征转换
        poly = PolynomialFeatures(degree=num_components)
        X_poly = poly.fit_transform(np.arange(len(self.data)).reshape(-1, 1))

        # 使用线性回归模型进行拟合
        model = LinearRegression()
        model.fit(X_poly, self.data)

        # 预测出趋势项
        trend = model.predict(X_poly)

        # 残差项是原数据与趋势项的差值
        residual = self.data - trend

        return trend, residual

    # 矩阵分解（使用奇异值分解SVD）
    def matrix_decomposition(self, rank=2):
        """
        矩阵分解：使用奇异值分解(SVD)进行时间序列分解
        :param rank: 奇异值分解的秩，决定了分解的组件数，默认为5
        :return: decomposed（分解结果）, residual（残差项）
        """
        # 将时间序列数据转化为二维矩阵，适用于SVD
        data_matrix = self.data.values.reshape(-1, 1)  # 假设是单变量时间序列

        # 将数据转换为具有两个特征的二维矩阵
        data_matrix = np.hstack([data_matrix, data_matrix])  # 复制数据列以满足SVD的要求

        # 使用TruncatedSVD（截断SVD）进行矩阵分解
        svd = TruncatedSVD(n_components=rank)
        decomposed = svd.fit_transform(data_matrix)

        # 重构数据
        reconstructed = svd.inverse_transform(decomposed)

        # 计算残差项
        residual = self.data - reconstructed[:, 0]  # 使用第一列的重构结果

        return decomposed, residual


# 示例用法
if __name__ == "__main__":
    # 读取ETTm2.csv数据集，将date列解析为日期时间类型并设置为索引
    file_path = 'chap2_basic_concept/data/ETTm2.csv'
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # 以单变量时序数据为例，选择一个字段作为时间序列数据，这里选择 'OT' 字段
    data = df['OT']

    # 创建TimeSeriesDecomposition对象
    ts_decomp = TimeSeriesDecomposition(data)

    # 经典分解法
    trend, seasonal, residual = ts_decomp.classical_decomposition(seasonal_periods=3)
    trend_df = trend.reset_index()
    trend_df.columns = ['date', 'trend']
    trend_df.to_csv('chap2_basic_concept/data/Decomposition_Results/classical_trend.csv', index=False)

    seasonal_df = seasonal.reset_index()
    seasonal_df.columns = ['date', 'seasonal']
    seasonal_df.to_csv('chap2_basic_concept/data/Decomposition_Results/classical_seasonal.csv', index=False)

    residual_df = residual.reset_index()
    residual_df.columns = ['date', 'residual']
    residual_df.to_csv('chap2_basic_concept/data/Decomposition_Results/classical_residual.csv', index=False)

    # 基函数扩展
    trend, residual = ts_decomp.basis_function_expansion(num_components=3)
    trend_series = pd.Series(trend, index=data.index, name='trend')
    trend_series.reset_index().to_csv('chap2_basic_concept/data/Decomposition_Results/basis_trend.csv', index=False)

    residual_df = residual.reset_index()
    residual_df.columns = ['date', 'residual']
    residual_df.to_csv('chap2_basic_concept/data/Decomposition_Results/basis_residual.csv', index=False)

    # 矩阵分解
    decomposed, residual = ts_decomp.matrix_decomposition(rank=2)
    decomposed_df = pd.DataFrame(decomposed, index=data.index, columns=['component_1', 'component_2'])
    decomposed_df.reset_index().to_csv('chap2_basic_concept/data/Decomposition_Results/matrix_decomposed.csv', index=False)

    residual_df = residual.reset_index()
    residual_df.columns = ['date', 'residual']
    residual_df.to_csv('chap2_basic_concept/data/Decomposition_Results/matrix_residual.csv', index=False)