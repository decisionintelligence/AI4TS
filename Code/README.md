# AI4TS

一个全面的时间序列分析基础代码库，涵盖数据预处理、预测、异常检测和分类，使用深度学习模型。

## 项目结构

```
AI4TS/Code/
├── README.md                          # 本说明文档
├── requirements.txt                   # 依赖包列表
├── chap2_basic_concept/               # 基础概念
│   ├── Decomposition_processing.py    # 时间序列分解
│   ├── Smoothing_processing.py        # 平滑技术
│   ├── Stabilization_processing.py    # 平稳化方法
│   ├── Missing_value_processing.py    # 缺失值处理
│   ├── Fourier_transform.py           # 傅里叶分析
│   └── data/                          # 数据文件
├── chap3_forecasting/                 # 时间序列预测
│   ├── Linear.py, CNN.py, RNN.py, TCN.py, Transformer.py
│   └── data/
├── chap4_anomaly_detection/           # 异常检测
│   ├── AE.py, CNN.py, RNN.py, TCN.py, Transformer.py, VAE.py
│   └── data/
└── chap5_classification/              # 时间序列分类
    ├── Linear.py, CNN.py, RNN.py, TCN.py, Transformer.py
    └── data/
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 数据集下载

- **ETTm2**: 用于预测和异常检测
  - 下载地址: [Google Drive](https://drive.google.com/file/d/1v5az7yXB5J4se5UHrmXzedSCrMlDWAtH/view?usp=sharing)
  - 放置位置: `chap2_basic_concept/data/` 和 `chap3_forecasting/data/`

- **ACSF1**: 用于分类（UCR时间序列分类基准）
  - 下载地址: [UCR Time Series Classification](https://timeseriesclassification.com/dataset.php)
  - 放置位置: `chap5_classification/data/ACSF1/`

### 运行示例
```bash
# 基础概念示例
python chap2_basic_concept/Decomposition_processing.py

# 预测模型示例
python chap3_forecasting/Linear.py

# 分类模型示例
python chap5_classification/Linear.py
```

## 主要功能

### 第2章：基础概念
- **分解**: 经典分解、基函数分解、矩阵分解
- **平滑**: 移动平均、指数平滑方法
- **平稳化**: 差分、对数变换、归一化技术
- **缺失值处理**: LOCF、NOCB、插值、统计插补
- **傅里叶分析**: 频域变换

### 第3章：预测模型
- **架构**: 线性模型、CNN、RNN、TCN、Transformer
- **数据集**: ETTm2电力变压器温度数据
- **任务**: 多步时间序列预测

### 第4章：异常检测
- **方法**: 基于自编码器的重构方法
- **模型**: AE、CNN-AE、RNN-AE、TCN-AE、Transformer-AE、VAE
- **检测**: 通过重构误差识别异常

### 第5章：分类
- **数据集**: UCR时间序列分类基准（ACSF1）
- **模型**: 线性、CNN、RNN、TCN、Transformer分类器
- **评估**: 标准分类指标

## 依赖包

- `torch` - 深度学习框架
- `numpy` - 数值计算
- `scikit-learn` - 机器学习工具
- `pandas` - 数据处理


---


