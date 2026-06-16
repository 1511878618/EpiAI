<picture>
  <source media="(prefers-color-scheme: dark)" srcset="">
  <img alt="EpiAI" src="">
</picture>

# EpiAI — 传染病爆发预测框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/xutingfeng/EpiAI/actions/workflows/pytest.yml/badge.svg)](https://github.com/xutingfeng/EpiAI/actions)

**EpiAI** 是一个端到端的传染病爆发预测框架，支持深度学习、机器学习
和传统时间序列模型。提供统一的数据管道、模型注册系统、训练入口、
多模型管理（ModelVault）和生产部署运行时（DeploymentRuntime）。

---

## 特性

- **统一数据管道** — CSV → 拆分 → 变换 → 滑窗，一行代码完成
- **三大模型族** — 深度学习 (PyTorch)、机器学习 (sklearn)、时间序列 (ARIMA/ETS)，统一训练接口
- **模型注册系统** — 29+ 个内置模型，通过 `@register` 一行扩展
- **ModelVault** — 多模型存储、对比、批量推理
- **DeploymentRuntime** — 生产级部署，统一数据表 + 时间连续性检查 + 持久化
- **爆发感知损失** — 专门为传染病爆发期设计的损失函数
- **多实体支持** — 单城市 / 多城市 / 按实体独立建模

---

## 特性

- **统一数据管道** — CSV → 拆分 → 变换 → 滑窗，一行代码完成
- **三大模型族** — 深度学习 (PyTorch)、机器学习 (sklearn)、时间序列 (ARIMA/ETS)
- **爆发感知损失** — 专门为传染病爆发期设计的损失函数
- **模型注册系统** — 29+ 个内置模型，通过 `@register` 扩展
- **推理部署** — 训练 → 保存 → 加载 → 新数据预测，完整链路
- **多实体支持** — 单城市 / 多城市 / 按实体独立建模

---

## 快速安装

```bash
git clone https://github.com/your-repo/EpiAI.git
cd EpiAI
pip install -e .                     # 基础安装
pip install -e ".[xgb,lgbm]"        # 加 sklearn 模型
pip install -e ".[all]"             # 全部（含 PyTorch）
```

---

## 三分钟入门

```python
from EpiAI.dataset import ForecastPipeline
from EpiAI.models.registry import get
from EpiAI.trainer import EpiAITrainer

# 1. 数据管道
bundle = ForecastPipeline.quick(
    path="data.csv",
    time_col="time",
    target_cols="cases",
    feature_cols=["temp", "humid"],
    lookback=12, horizon=3,
)

# 2. 训练（一键切换模型）
model = get("RF")(input_dim=bundle.n_features,
                  lookback=12, horizon=3, target_dim=1)
result = EpiAITrainer(model=model).fit(bundle)

print(result.metrics)                     # MAE / RMSE / R²
```

---

## 核心概念

### 数据管道

```
CSV / Feather → TimeSeriesData
  → SplitStrategy（按时间/按实体/自定义）
    → Transform（对数/标准化/时间特征/滞后）
      → SlidingWindow → PipelineBundle（3D 数组）
```

```python
pipeline = ForecastPipeline(
    loader=CsvLoader(time_col="date", target_cols="cases",
                     feature_cols=["feature1", "feature2"]),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=Compose([
        Log1pTransform(columns=["cases"]),
        StandardScaler(),
        DateFeatures(time_col="date", features=["month", "season"]),
        FeatureLag(columns=["feature1"], lags=[1, 3, 6]),
    ]),
    window=SlidingWindow(lookback=12, horizon=3),
)
bundle = pipeline.run("data.csv")
```

### 三大模型族

| 族 | Paradigm | 训练方式 | 示例模型 |
|----|----------|---------|---------|
| Torch | `"torch"` | 梯度下降，GPU | CNN, LSTM, Transformer, TimesNet |
| Sklearn | `"sklearn"` | 一次 fit | XGBoost, LightGBM, RandomForest, SVR |
| TimeSeries | `"ts"` | 统计推断 | ARIMA, ETS |

> 三种模型的训练接口完全统一：

```python
# Torch
EpiAITrainer(model=get("LSTM")(...), loss=OutbreakAwareLoss(...)).fit(bundle)

# Sklearn — loss/optimizer 自动忽略
EpiAITrainer(model=get("XGB")(...)).fit(bundle)

# TimeSeries — 自动走 fit_sequence / rolling origin
EpiAITrainer(model=get("ETS")(seasonal_periods=12)).fit(bundle)
```

### 模型管理与部署

```python
# 单模型推理
inferer = InferencePipeline.from_train_result(result)
pred = inferer.predict(new_data_df)     # 对新数据预测
inferer.save("model.zip")
inferer = InferencePipeline.load("model.zip")

# 多模型管理
vault = ModelVault.from_results({"RF": r_rf, "XGB": r_xgb}, bundle)
vault.save("/tmp/vault/")
vault.summary()                         # 对比表
vault.best("R2")                        # 选最优模型

# 生产部署
runtime = DeploymentRuntime(vault, time_col="time")
runtime.data_table = training_data       # 加载历史数据
result = runtime.feed(new_data)          # 自动检查时间连续性并预测
runtime.save("/tmp/runtime/")
```

---

## 模型一览

### 深度学习（10 个）

`MLP` · `LSTM` · `CNN` · `CNN-LSTM` · `ResNet` · `TCN` · `Transformer` · `DLinear` · `Autoformer` · `TimesNet`

### 机器学习（6 个）

`XGBoost` · `LightGBM` · `RandomForest` · `SVR` · `LinearReg` · `TabPFN`

### 时间序列（2 个）

`ARIMA` · `ETS`

```python
from EpiAI.models.registry import list_models

list_models()               # 全部 29+ 个别名
list_models("torch")        # 仅深度学习
list_models("sklearn")      # 仅机器学习
list_models("ts")           # 仅时间序列
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [概述](docs/index.md) | 功能清单、模型一览、文档索引 |
| [快速上手](docs/quickstart.md) | 5 分钟跑通完整流程 |
| [框架架构](docs/architecture.md) | 整体分层、设计原则、文件结构 |
| [数据管道](docs/data-pipeline.md) | 加载 / 拆分 / 变换 / 滑窗 |
| [数据模块指南](docs/guides/dataset.md) | 数据功能详解 + 添加新 Transform / Split |
| [模型模块指南](docs/guides/models.md) | 注册系统 + 添加新模型 |
| [训练模块指南](docs/guides/training.md) | 训练器 + 添加新范式 |
| [部署模块指南](docs/guides/deployment.md) | 推理 / 部署 / 扩展 |
| [部署 API](docs/api-deployment.md) | DeploymentRuntime 接口说明 |
| [快速教程](tutorial/tutorial-dengue.ipynb) | 登革热快速体验 |
| [完整教程](tutorial/tutorial-full.ipynb) | 完整流程：留数据 → 训练 → 逐月 feed → 对比 |
| [更新日志](CHANGELOG.md) | 版本历史 |
| [贡献指南](CONTRIBUTING.md) | 开发环境、代码风格、PR 流程 |

---

## 扩展指南

### 添加新模型

```python
from EpiAI.models.base import SklearnMixin
from EpiAI.models.registry import register

@register("MyModel", "my_model")
class MyForecaster(SklearnMixin):
    def fit(self, train_x, train_y, val_x=None, val_y=None):
        ...
    def predict(self, x) -> np.ndarray:
        ...
```

无需修改任何其他文件。

### 添加新变换

```python
from EpiAI.dataset.base import Transform

class MyTransform(Transform):
    def transform(self, df):
        return do_something(df)
```

### 添加新拆分策略

```python
from EpiAI.dataset.base import SplitStrategy

class MySplit(SplitStrategy):
    def split(self, data):
        return SplitResult(...)
```

---

## 依赖

| 依赖 | 用途 | 必需 |
|------|------|------|
| `pandas`, `numpy`, `scikit-learn` | 数据处理 | ✅ |
| `torch` | 深度学习模型 | ❌ 可选 |
| `xgboost`, `lightgbm` | 树模型 | ❌ 可选 |
| `pmdarima` | ARIMA | ❌ 可选 |
| `statsmodels` | ETS | ❌ 可选 |
| `tabpfn` | TabPFN | ❌ 可选 |

---

## 开源协议

MIT License

---

## 引用

```bibtex
@software{epiai2025,
  author = {Xu, Tingfeng},
  title = {EpiAI: End-to-end Infectious Disease Forecasting Framework},
  year = {2025},
  url = {https://github.com/your-repo/EpiAI}
}
```
