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
    # 生成一个示例时间序列数据
    data = pd.Series([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987])

    # 创建TimeSeriesDecomposition对象
    ts_decomp = TimeSeriesDecomposition(data)

    # 经典分解法
    trend, seasonal, residual = ts_decomp.classical_decomposition(seasonal_periods=3)
    print("经典分解法（加性模型）:")
    print("趋势项:\n", trend)
    print("季节项:\n", seasonal)
    print("残差项:\n", residual)

    # 基函数扩展
    trend, residual = ts_decomp.basis_function_expansion(num_components=3)
    print("\n基函数扩展:")
    print("趋势项:\n", trend)
    print("残差项:\n", residual)

    # 矩阵分解
    decomposed, residual = ts_decomp.matrix_decomposition(rank=2)
    print("\n矩阵分解 (SVD):")
    print("分解后的结果:\n", decomposed)
    print("残差项:\n", residual)
