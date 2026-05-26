# EpiAI

End-to-end infectious disease outbreak forecasting framework with deep learning and tabular models.

## Features

- **Multi-target Time Series Forecasting**: City-by-city disease prediction with multiple targets
- **Rich Model Zoo**: 
  - PyTorch Models: CNN, LSTM, CNN-LSTM, ResNet, TCN, Transformer, DLinear, Autoformer, TimesNet
  - Tabular Models: LightGBM, XGBoost, TabPFN
- **Outbreak-Aware Loss Functions**: Specialized losses for imbalanced outbreak periods
- **Flexible Data Pipeline**: Sliding window generation, normalization, train/val/test splitting
- **Advanced Attention Layers**: AutoCorrelation, Transformer EncDec, Crossformer, Pyraformer, etc.

## Installation

```bash
# From source
pip install -e .

# With all dependencies
pip install -e ".[all]"

# With specific backend
pip install -e ".[lgbm]"   # LightGBM support
pip install -e ".[xgb]"     # XGBoost support
```

## Quick Start

### Data Preparation

```python
from disease_forecasting import (
    DatasetConfig,
    MultiTargetCityDatasetBuilder,
    ForecastDataModule,
)

config = DatasetConfig(
    data_path="data/Align_data_tensor_with_name.pt",
    target_feature_names=["登革热", "流感"],
    train_val_test_cutoff_line=(20, 27),
)

builder = MultiTargetCityDatasetBuilder(config)
bundle = builder.build()

# Use with PyTorch Lightning
datamodule = ForecastDataModule(bundle, batch_size=32)
```

### Model Training

```python
from models.torch_models import CNNForecaster, LSTMForecaster
from losses import OutbreakAwareLoss

# PyTorch model
model = CNNForecaster(
    input_len=14,
    pred_len=7,
    input_dim=10,
    hidden_dims=[64, 128],
    output_dim=2,
)

# Outbreak-aware loss
criterion = OutbreakAwareLoss(outbreak_threshold=100.0, outbreak_weight=5.0)
```

### Using Tabular Models

```python
from models.tabular_models import LGBMSingleForecaster

model = LGBMSingleForecaster(
    input_len=14,
    pred_len=7,
    n_estimators=100,
    learning_rate=0.05,
)
```

## Project Structure

```
EpiAI/
├── src/EpiAI/
│   ├── __init__.py
│   ├── losses.py           # Loss functions (outbreak-aware, trend-aware, etc.)
│   ├── utils.py
│   ├── dataset/            # Data processing pipeline
│   │   ├── builder.py      # MultiTargetCityDatasetBuilder
│   │   ├── config.py       # DatasetConfig
│   │   ├── containers.py   # Data containers
│   │   ├── datamodule.py    # PyTorch Lightning DataModule
│   │   ├── inspector.py     # Dataset inspection utilities
│   │   ├── normalizer.py   # Data normalization
│   │   ├── splitter.py     # Train/val/test splitting
│   │   ├── task_builder.py # Feature engineering
│   │   ├── windowing.py    # Sliding window generation
│   │   └── io.py           # Data loading
│   ├── layers/             # Advanced neural network layers
│   │   ├── AutoCorrelation.py
│   │   ├── Autoformer_EncDec.py
│   │   ├── Transformer_EncDec.py
│   │   ├── Crossformer_EncDec.py
│   │   ├── Pyraformer_EncDec.py
│   │   ├── ETSformer_EncDec.py
│   │   ├── SelfAttention_Family.py
│   │   ├── FourierCorrelation.py
│   │   ├── MultiWaveletCorrelation.py
│   │   ├── MSGBlock.py
│   │   ├── MambaBlock.py
│   │   ├── DWT_Decomposition.py
│   │   └── Embed.py
│   └── models/
│       ├── torch_models/   # PyTorch forecasting models
│       │   ├── cnn.py
│       │   ├── lstm.py
│       │   ├── cnn_lstm.py
│       │   ├── resnet.py
│       │   ├── tcn.py
│       │   ├── transformer.py
│       │   ├── dlinear.py
│       │   ├── Autoformer.py
│       │   └── TimesNet.py
│       └── tabular_models/ # Traditional ML models
│           ├── lgbm.py
│           ├── xgb.py
│           └── tabpfn.py
└── tests/
```

## Available Loss Functions

| Loss Function | Description |
|--------------|-------------|
| `MAPELoss` | Mean Absolute Percentage Error |
| `SMAPELoss` | Symmetric MAPE |
| `LogCoshLoss` | Log-Cosh loss |
| `CorrelationLoss` | Waveform consistency loss |
| `MultiQuantileLoss` | Joint quantile loss |
| `TrendAwareLoss` | MAE + trend consistency |
| `OutbreakAwareLoss` | Weighted MAE for outbreak periods |
| `AsymmetricOutbreakLoss` | Penalizes underestimation during outbreaks |
| `OutbreakWeightedHuberLoss` | Huber + outbreak + trend |
| `FocalRegressionLoss` | Focal-style modulated regression |
| `RegressionWithOutbreakBCELoss` | Combined regression + classification |

## Available Models

### PyTorch Models

| Model | Description |
|-------|-------------|
| `CNNForecaster` | CNN-based forecasting |
| `LSTMForecaster` | LSTM-based forecasting |
| `CNNLSTMForecaster` | CNN + LSTM hybrid |
| `ResNetForecaster` | ResNet-style forecasting |
| `TCNForecaster` | Temporal Convolutional Network |
| `TransformerForecaster` | Vanilla Transformer |
| `DLinearForecaster` | DLinear decomposition |
| `AutoformerForecaster` | Autoformer model |
| `TimesNetForecaster` | TimesNet model |

### Tabular Models

| Model | Description |
|-------|-------------|
| `LGBMSingleForecaster` | LightGBM single-step |
| `XGBSingleForecaster` | XGBoost single-step |
| `TabPFNMultiForecaster` | TabPFN multi-step |

## License

MIT License
