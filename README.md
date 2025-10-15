# AI4TS - AI for Time Series

A comprehensive time series analysis project covering data preprocessing, forecasting, anomaly detection, and classification using deep learning models.

## Project Structure

```
AI4TS/
├── code/
│   ├── chap2_basic_concept/          # Basic Concepts
│   │   ├── Decomposition_processing.py     # Time series decomposition
│   │   ├── Smoothing_processing.py         # Smoothing techniques
│   │   ├── Stabilization_processing.py     # Stationarization methods
│   │   ├── Missing_value_processing.py     # Missing value imputation
│   │   ├── Fourier_transform.py            # Fourier analysis
│   │   └── data/
│   ├── chap3_forecasting/             # Time Series Forecasting
│   │   ├── Linear.py, CNN.py, RNN.py, TCN.py, Transformer.py
│   │   └── data/
│   ├── chap4_anomaly_detection/       # Anomaly Detection
│   │   ├── AE.py, CNN.py, RNN.py, TCN.py, Transformer.py, VAE.py
│   │   └── data/
│   ├── chap5_classification/          # Time Series Classification
│   │   ├── Linear.py, CNN.py, RNN.py, TCN.py, Transformer.py
│   │   └── data/
├── requirements.txt
└── README.md
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Datasets
- **ETTm2**: For forecasting and anomaly detection
  - Download from: [Google Drive](https://drive.google.com/file/d/1v5az7yXB5J4se5UHrmXzedSCrMlDWAtH/view?usp=sharing)
  - Place in: `code/chap2_basic_concept/data/` and `code/chap3_forecasting/data/`

- **ACSF1**: For classification (UCR archive)
  - Download from: [UCR Time Series Classification](https://timeseriesclassification.com/dataset.php)
  - Place in: `code/chap5_classification/data/ACSF1/`

### Run Examples
```bash
# Basic concepts
python code/chap2_basic_concept/Decomposition_processing.py

# Forecasting models
python code/chap3_forecasting/Linear.py

# Classification models  
python code/chap5_classification/Linear.py
```

## Key Features

### Chapter 2: Basic Concepts
- **Decomposition**: Classical, basis function, matrix decomposition
- **Smoothing**: Moving averages, exponential smoothing methods
- **Stationarization**: Differencing, log transform, normalization techniques
- **Missing Values**: LOCF, NOCB, interpolation, statistical imputation
- **Fourier Analysis**: Frequency domain transformation

### Chapter 3: Forecasting Models
- **Architectures**: Linear, CNN, RNN, TCN, Transformer
- **Dataset**: ETTm2 electricity transformer temperature data
- **Task**: Multi-step time series forecasting

### Chapter 4: Anomaly Detection
- **Approach**: Reconstruction-based using autoencoders
- **Models**: AE, CNN-AE, RNN-AE, TCN-AE, Transformer-AE, VAE
- **Detection**: Identify anomalies through reconstruction error

### Chapter 5: Classification
- **Dataset**: UCR time series classification benchmark (ACSF1)
- **Models**: Linear, CNN, RNN, TCN, Transformer classifiers
- **Evaluation**: Standard classification metrics

## Dependencies

- `torch` - Deep learning framework
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation

## Contributing

Contributions welcome! Please fork the repository and create a pull request.
