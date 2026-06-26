<picture>
  <source media="(prefers-color-scheme: dark)" srcset="">
  <img alt="EpiAI" src="">
</picture>

# EpiAI — 传染病爆发预测框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/badge/pypi-v0.7.0-blue.svg)](https://pypi.org/project/EpiAI/)

**EpiAI** 是一个端到端的传染病爆发预测框架，专为疫情监测与预警设计。
支持 **22 个内置模型**（深度学习 + 机器学习 + 时间序列），提供从数据处理、模型训练、多模型管理到生产部署和风险预警的完整链路。

---

## 特性

- **22 个内置模型** — 6 个时间序列（ARIMA/ETS/Prophet/BSTS/STLM/Serfling）、6 个机器学习（RF/XGB/LGBM/SVR/LinearReg/TabPFN）、10 个深度学习（LSTM/CNN/TCN/ResNet/Autoformer/TimesNet/Transformer/MLP/CNN-LSTM/DLinear），统一训练接口
- **统一数据管道** — CSV → 拆分 → 变换 → 滑窗，一行代码完成
- **模型注册系统** — 通过 `@register` 一行扩展，无需修改框架代码
- **ModelVault** — 多模型存储、对比（Pearson r 排序）、批量推理
- **DeploymentRuntime** — 生产部署：统一 data_table、自动 gap 补齐、滑动窗口重训、持久化
- **风险预警模块** — 独立的风险评估（分位数/阈值/Z-score/环比），多模型融合+预警报告
- **Transform 透明化** — 模型内部统一处理归一化/逆归一化，用户只接触原始尺度数据
- **全 GPU 支持** — XGB/LightGBM (GPU) + PyTorch (CUDA)

---

## 快速安装

```bash
git clone https://github.com/xutingfeng/EpiAI.git
cd EpiAI
pip install -e .                          # 基础安装（sklearn 模型）
pip install -e ".[torch]"                 # 加深度学习
pip install -e ".[xgb,lgbm,prophet]"      # 加树模型 + Prophet
pip install -e ".[all]"                   # 全部
```

---

## 三分钟入门

```python
from EpiAI.dataset import ForecastPipeline
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import ModelVault, DeploymentRuntime
import pandas as pd

# ── 1. 数据管道 ────────────────────────────────────
bundle = ForecastPipeline.quick(
    path="data.csv",
    time_col="time",
    target_cols="cases",
    lookback=12, horizon=3,
)
print(f"训练样本: {bundle.train_x.shape}")

# ── 2. 训练多个模型 ────────────────────────────────
results = {}
for model_name in ("ETS", "RF", "LSTM"):
    model = get(model_name)(
        input_dim=bundle.n_features,
        lookback=12, horizon=3, target_dim=1,
        **(dict(seasonal_periods=12) if model_name == "ETS" else {}),
    )
    r = EpiAITrainer(model=model, verbose=False).fit(bundle)
    results[model_name] = r
    print(f"{model_name:5s} Pearson r = {r.metrics['PearsonR']['cases']:.3f}")

# ── 3. 多模型管理 ──────────────────────────────────
vault = ModelVault.from_results(results, bundle)
vault.summary()          # 对比表
print(f"最佳模型: {vault.best('PearsonR')}")

# ── 4. 生产部署 + 预测 ────────────────────────────
history = pd.concat([bundle.train_df, bundle.val_df, bundle.test_df])
runtime = DeploymentRuntime(vault=vault, data_table=history)
runtime.update_ts()                     # TS 模型追到最新
preds = runtime.predict(horizon=12)     # pd.DataFrame
print(preds)

# ── 5. 风险预警 ────────────────────────────────────
from EpiAI.risk import RiskScorer, WarningRule
scorer = RiskScorer(method="quantile").fit(runtime.data_table)
risk_df = scorer.score_df(preds)
report = WarningRule(ensemble="max").evaluate(risk_df)
print(WarningRule(ensemble="max").report(report))
```

---

## 三大模型族

| 族 | 范式 | 训练方式 | 模型数 |
|----|------|---------|--------|
| **时间序列** | `"ts"` | 统计推断，fit_sequence + forecast | 6 |
| **机器学习** | `"sklearn"` | 批量 fit，滑窗输入 | 6 |
| **深度学习** | `"torch"` | 梯度下降，GPU/CPU，滑窗输入 | 10 |

训练接口完全统一：

```python
# 时间序列 — 自动走 fit_sequence / rolling origin
EpiAITrainer(model=get("ETS")(seasonal_periods=12)).fit(bundle)

# 机器学习 — optimizer 自动忽略
EpiAITrainer(model=get("XGB")(...)).fit(bundle)

# 深度学习 — AdamW + early stopping
EpiAITrainer(model=get("LSTM")(...)).fit(bundle)
```

---

## 模型一览

### 时间序列（6 个）

| 模型 | 引擎 | 适用场景 |
|------|------|---------|
| **ARIMA** | pmdarima auto_arima | 通用 baseline，自动选阶 |
| **ETS** | statsmodels ExponentialSmoothing | 快速 baseline，训练毫秒级 |
| **Prophet** | Facebook Prophet | 趋势变化点、缺值容忍 |
| **BSTS** | PyMC MCMC | 贝叶斯推断、后验置信区间 |
| **Serfling** | sklearn 线性回归 | 超额估计，经典流行病学方法 |
| **STLM** | statsmodels STL + pm.ARIMA | 复杂季节模式 |

### 机器学习（6 个）

| 模型 | 引擎 | 特性 |
|------|------|------|
| **RF** | sklearn RandomForest | 原生多输出，抗过拟合 |
| **XGB** | xgboost XGBRegressor | 原生多输出，GPU 加速 |
| **LGBM** | lightgbm LGBMRegressor | GPU 加速，leaf-wise 树 |
| **SVR** | sklearn SVR | 小数据泛化好 |
| **LinearReg** | sklearn LinearRegression | 极快，可解释 |
| **TabPFN** | TabPFN | 预训练 Transformer，Few-shot |

### 深度学习（10 个）

| 模型 | 结构特性 | 适用场景 |
|------|---------|---------|
| **LSTM** | LSTM → Dropout → FC | 时序通用，中等数据量 |
| **CNN** | Conv1d × 2 → MaxPool → FC | 快速特征提取 |
| **CNN-LSTM** | Conv1d → BN → LSTM → FC | 局部特征 + 长期依赖 |
| **MLP** | FC → ReLU → Dropout × 2 → FC | 极简 baseline |
| **ResNet** | Conv1d → BN → ReLU × blocks → FC | 残差稳定训练 |
| **TCN** | 空洞因果卷积 + BN + 残差 | 替代 RNN 的时序建模 |
| **Transformer** | PositionalEncoding + Encoder | 全局 attention |
| **Autoformer** | AutoCorrelation + 序列分解 | 长期预测 |
| **TimesNet** | 1D→2D 时序 + Inception 模块 | 多周期叠加 |
| **DLinear** | 序列分解 + 线性层 | 零超参 baseline |

---

## 部署流程

```python
from EpiAI.inference import ModelVault, DeploymentRuntime

# 加载
vault = ModelVault.load("/path/to/vault/")
runtime = DeploymentRuntime(
    vault=vault, data_table=history_df,
    time_col="time", time_unit="MS",
)

# 预测未来 12 个月
preds = runtime.predict(horizon=12)

# 新数据到达 – 更新 TS 模型状态
runtime.feed(new_month_data)
runtime.update_ts()           # 滑动窗口重训
preds = runtime.predict(horizon=6)

# 持久化
runtime.save("/tmp/runtime/")
runtime = DeploymentRuntime.load("/tmp/runtime/")
```

核心设计：
- **data_table** 存原始尺度数据，模型内部 `InferencePipeline` 透明处理 transform/inverse
- **TS 模型**自动 gap 补齐 — `forecast(gap + horizon)` → 取最后 horizon 步，保证所有模型时间对齐
- **update_ts** — 保持原始训练数据量，滑动窗口到最新

---

## 风险预警模块

```python
from EpiAI.risk import RiskScorer, WarningRule

# 4 种评分方法
scorer = RiskScorer(method="quantile").fit(history)
risk_df = scorer.score_df(predictions)

# 3 种融合策略
rule = WarningRule(ensemble="max")
report = rule.evaluate(risk_df)
print(rule.report(report))
```

| 组件 | 说明 |
|------|------|
| `RiskScorer` | 分位数 / 阈值 / Z-score / 环比 → 风险等级 0-3 |
| `WarningRule` | max / mean / consensus 融合 + 连续高风险升级检测 |

---

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户代码                                   │
│  ForecastPipeline → EpiAITrainer → ModelVault → DeploymentRuntime │
├─────────────────────────────────────────────────────────────┤
│  数据层                       模型层                   部署层│
│  CsvLoader        @register("LSTM")              InferencePipeline │
│  TimeSplit        LSTMForecaster(TorchMixin)     ModelVault │
│  Compose          EpiAITrainer.fit(bundle)       DeploymentRuntime │
│  SlidingWindow                                      RiskScorer │
└─────────────────────────────────────────────────────────────┘
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [快速上手](docs/quickstart.md) | 5 分钟跑通完整流程 |
| [框架架构](docs/architecture.md) | 整体分层、设计原则 |
| [数据管道](docs/data-pipeline.md) | 加载 / 拆分 / 变换 / 滑窗 |
| [数据模块指南](docs/guides/dataset.md) | 添加新 Transform / Split |
| **模型清单** (docs/guides/models.md) | 22 个模型的输入/输出/场景 |
| [训练指南](docs/guides/training.md) | 训练器配置 |
| [部署指南](docs/guides/deployment.md) | 推理 / 部署 / 迁移 |
| [部署 API](docs/api-deployment.md) | DeploymentRuntime 接口 |
| [快速教程](tutorial/tutorial-dengue.ipynb) | 登革热：22 模型对比 + 部署 |
| [完整教程](tutorial/tutorial-full.ipynb) | 完整流程 |
| [更新日志](CHANGELOG.md) | 版本历史 |

---

## 扩展指南

### 添加新模型

```python
from EpiAI.models.base import SklearnMixin
from EpiAI.models.registry import register

@register("MyModel")
class MyForecaster(SklearnMixin):
    def fit(self, train_x, train_y, val_x=None, val_y=None):
        ...
    def predict(self, x):
        ...
```

无需修改任何其他文件。

### 添加新变换 / 拆分

```python
from EpiAI.dataset.base import Transform, SplitStrategy
```

继承 `Transform` 或 `SplitStrategy` 即可注册。

---

## 依赖

| 依赖 | 用途 | 必需 |
|------|------|------|
| `pandas`, `numpy` | 数据处理 | ✅ |
| `scikit-learn` | 基础模型 / 评估 | ✅ |
| `torch` | 深度学习 (LSTM/CNN/TCN/...) | ❌ `[torch]` |
| `xgboost` | XGB 模型 | ❌ `[xgb]` |
| `lightgbm` | LGBM 模型 | ❌ `[lgbm]` |
| `pmdarima` | ARIMA | ❌ `[ts]` |
| `statsmodels` | ETS / STLM | ❌ `[ts]` |
| `prophet` | Prophet | ❌ `[prophet]` |
| `pymc` | BSTS (MCMC) | ❌ `[pymc]` |
| `tabpfn` | TabPFN | ❌ `[tabpfn]` |

```bash
pip install EpiAI                      # 基础（sklearn 模型）
pip install EpiAI[torch]               # 加深度学习
pip install EpiAI[xgb,lgbm,ts]         # 加树模型 + 时序
pip install EpiAI[all]                 # 全部
```
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
  url = {https://github.com/xutingfeng/EpiAI}
}
```
