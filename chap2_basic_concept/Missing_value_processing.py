import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
from statsmodels.tsa.arima.model import ARIMA
from sklearn.mixture import GaussianMixture

#1.删除法
# 删除缺失值的方法
def RemoveMissingValues(df, axis=0):
    """
    删除缺失值的方法
    :param df: 含缺失值的时序数据
    :param axis: 0 删除含缺失值的行，1 删除含缺失值的列
    :return: 删除后的 DataFrame
    """
    if axis == 0:
        return df.dropna(axis=0)  # 删除含缺失值的行
    elif axis == 1:
        return df.dropna(axis=1)  # 删除含缺失值的列
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")


# 2.填充法
# 就近填充法
def LOCF(series):
    """
    前推法：用缺失值的前一个值填补缺失值
    """
    return series.ffill()

def NOCB(series):
    """
    后推法：用缺失值的后一个值填补缺失值
    """
    return series.bfill()

# 特征值填充法
def Statistics(series, method="mean"):
    """
    特征值填充法
    :param series: 含缺失值的时序数据
    :param method: 填充方法，可选"mean"（均值）、"median"（中值）、"mode"（众数）
    :return: 填补后的时序
    """
    if method == "mean":
        return series.fillna(series.mean())
    elif method == "median":
        return series.fillna(series.median())
    elif method == "mode":
        return series.fillna(series.mode()[0])
    else:
        raise ValueError("method must be 'mean', 'median', or 'mode'")

# 线性插值法
def Linear(series):
    """
    线性插值法：通过插值预测缺失值
    """
    return series.interpolate(method="linear")


# 基于统计模型的方法，以 "ARIMA"、"EM"方法为例
def Statistical_Model(series, model="ARIMA", **kwargs):
    """
    基于统计模型的方法
    :param series: 含缺失值的时序数据
    :param model: 使用的模型类型（支持"ARIMA"、"EM"等）
    :return: 填补后的时序
    """
    if model == "ARIMA":
        filled_series = series.copy()
        nan_indices = np.where(filled_series.isna())[0]

        training_data = filled_series.dropna()
        arima_model = ARIMA(training_data, order=kwargs.get("order", (1, 1, 1)))
        fitted_model = arima_model.fit()

        for idx in nan_indices:
            forecast = fitted_model.forecast(steps=1)
            filled_series.iloc[idx] = forecast.iloc[0]
            training_data = pd.concat([training_data, pd.Series([forecast.iloc[0]], index=[idx])])
            fitted_model = ARIMA(training_data, order=kwargs.get("order", (1, 1, 1))).fit()

        return filled_series

    elif model == "EM":
        # 使用EM算法进行缺失值填补
        filled_series = series.copy()
        # 使用高斯混合模型（EM算法的实现之一）来估计缺失值
        gmm = GaussianMixture(n_components=2, random_state=42)
        training_data = filled_series.dropna().values.reshape(-1, 1)
        gmm.fit(training_data)

        # 对缺失值进行填补
        nan_indices = filled_series[filled_series.isna()].index
        for idx in nan_indices:
            # 对于每个缺失值，预测其值
            filled_series.loc[idx] = gmm.sample(1)[0][0]

        return filled_series

    else:
        raise ValueError("Unsupported model: " + model)

# 针对多变量时序的缺失值填补方法，以 SVD、NMF 方法为例
def Statistical_Multi_Variable(data, method="SVD"):
    # 如果选择方法是 "SVD"
    if method == "SVD":
        df = data.copy()
        
        # 步骤1：用列均值填充缺失值
        df_filled = df.copy()
        df_filled.fillna(df.mean(), inplace=True)
        
        # 步骤2：对填充后的数据应用 SVD（奇异值分解）
        matrix_filled = df_filled.values
        U, S, V = np.linalg.svd(matrix_filled, full_matrices=False)
        
        # 步骤3：使用 SVD 组件重构数据
        df_svd = np.dot(U, np.dot(np.diag(S), V))
        df_svd = pd.DataFrame(df_svd, index=df.index, columns=df.columns)
        
        # 步骤4：使用 SVD 重构后的数据填补缺失值
        df_svd_filled = df.copy()
        df_svd_filled[data.isna()] = df_svd[data.isna()]
        
        # 步骤5：返回填充后的 DataFrame
        return df_svd_filled

    # 如果选择方法是 "NMF"
    elif method == "NMF":
        df = data.copy()
        
        # 步骤1：用列均值填充缺失值
        df_filled = df.copy()
        df_filled.fillna(df.mean(), inplace=True)
        
        # 步骤2：对填充后的数据应用 NMF（非负矩阵分解）
        nmf = NMF(n_components=min(df.shape) - 1, init='random', random_state=42)
        W = nmf.fit_transform(df_filled)
        H = nmf.components_
        
        # 步骤3：使用 NMF 组件重构数据
        df_nmf = np.dot(W, H)
        df_nmf = pd.DataFrame(df_nmf, index=df.index, columns=df.columns)
        
        # 步骤4：使用 NMF 重构后的数据填补缺失值
        df_nmf_filled = df.copy()
        df_nmf_filled[data.isna()] = df_nmf[data.isna()]
        
        # 步骤5：返回填充后的 DataFrame
        return df_nmf_filled

    # 如果输入的 method 不是 "SVD" 或 "NMF"，则抛出错误
    else:
        raise ValueError(f"Unsupported method: {method}")


# 基于机器学习的方法，以 KNN 方法为例
def Similarity(df, n_neighbors=5):
    """
    基于相似性的方法
    :param df: 含缺失值的多变量时序数据
    :param n_neighbors: KNN 的邻居数
    :return: 填补后的多变量时序
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_filled = imputer.fit_transform(df)
    return pd.DataFrame(df_filled, index=df.index, columns=df.columns)


# 示例用法
if __name__ == "__main__":
    # 生成一个示例时序数据
    data = pd.Series([1, np.nan, 3, np.nan, 5, 6, np.nan, 8])
    print("原始数据:\n", data)

    # 使用前推法
    print("LOCF:\n", LOCF(data))

    # 使用后推法
    print("NOCB:\n", NOCB(data))

    # 使用均值填充
    print("Mean Imputation:\n", Statistics(data, method="mean"))

    # 使用中值填充
    print("Median Imputation:\n", Statistics(data, method="median"))

    # 使用线性插值法
    print("Linear Interpolation:\n", Linear(data))

    # 使用统计模型（ARIMA）填补
    print("Statistical Model (ARIMA):\n", Statistical_Model(data, model="ARIMA", order=(1, 1, 1)))

    # 使用统计模型（EM）填补
    print("Statistical Model (EM):\n", Statistical_Model(data, model="EM"))

    # 示例多变量数据
    multi_data = pd.DataFrame({
        "var1": [1, np.nan, 3, 4],
        "var2": [2, 3, np.nan, 5],
        "var3": [np.nan, 2, 3, 4]
    })
    print("\n原始多变量数据:\n", multi_data)

    # 使用多变量填充（SVD）
    print("Statistical Multi-Variable (SVD):\n", Statistical_Multi_Variable(multi_data, method="SVD"))

    # 使用多变量填充（NMF）
    print("Statistical Multi-Variable (NMF):\n", Statistical_Multi_Variable(multi_data, method="NMF"))

    # 使用 KNN 方法填补
    print("KNN Imputation:\n", Similarity(multi_data))

    # 使用删除法删除含缺失值的行
    print("\nRemove Missing Values (Rows):\n", RemoveMissingValues(multi_data, axis=0))

    # 使用删除法删除含缺失值的列
    print("Remove Missing Values (Columns):\n", RemoveMissingValues(multi_data, axis=1))
