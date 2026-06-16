# 模型模块指南

## 功能介绍

模型模块位于 `EpiAI.models`，提供统一的模型注册系统和三族模型接口。

### 注册系统

所有模型通过 `@register` 装饰器注册到全局注册表，通过 `get()` 按名称获取。

```python
from EpiAI.models.registry import get, list_models

# 获取模型类
model_cls = get("RF")
model = model_cls(input_dim=8, lookback=12, horizon=3, target_dim=1)

# 列出可用模型
list_models()            # 全部 29+ 个
list_models("torch")     # 仅深度学习
list_models("sklearn")   # 仅机器学习
list_models("ts")        # 仅时间序列
```

### 三族接口

所有模型继承 `BaseForecaster`。EpiAITrainer 根据 `paradigm` 自动选择训练路径。

```python
model.paradigm()  # 返回 "torch" / "sklearn" / "ts"
```

**TorchMixin**（深度学习）

```python
@register("LSTM", "lstm")
class LSTMForecaster(nn.Module, TorchMixin):
    def __init__(self, input_dim, lookback, horizon, target_dim, ...): ...
    def forward(self, x): ...                      # 模型前向计算
    def predict(self, x): ...                      # 推理（默认 no_grad + 自动设备）
```

**SklearnMixin**（机器学习）

```python
@register("XGB", "xgboost", "XGBoost")
class XGBSingleForecaster(SklearnMixin):
    def __init__(self, input_dim, lookback, horizon, target_dim, xgb_params=None): ...
    def fit(self, train_x, train_y, val_x=None, val_y=None): ...
    def predict(self, x): ...
```

内置辅助方法：`_flatten_x()`、`_prepare_y()`、`_reshape_pred()`。

**TSMixin**（时间序列）

```python
@register("ETS", "ets")
class ETSForecaster(TSMixin):
    def __init__(self, seasonal_periods=12, seasonal="add", trend="add"): ...
    def fit_sequence(self, y_train, X_train=None): ...
    def predict_sequence(self, y_test, X_test=None, update_state=True): ...
    def forecast(self, n_periods, X_future=None): ...
```

---

## 可用模型清单

### 深度学习（10 个）

| 模型名 | 全称 |
|--------|------|
| `MLP` | 多层感知机 |
| `LSTM` | 长短期记忆网络 |
| `CNN` | 卷积神经网络 |
| `CNN-LSTM` | CNN + LSTM 混合 |
| `ResNet` | 残差网络 |
| `TCN` | 时序卷积网络 |
| `Transformer` | 自注意力模型 |
| `DLinear` | 线性分解 |
| `Autoformer` | 自相关机制 |
| `TimesNet` | 时序块 |

### 机器学习（6 个）

| 模型名 | 全称 |
|--------|------|
| `RF` | 随机森林 |
| `XGB` | XGBoost |
| `LGBM` | LightGBM |
| `SVR` | 支持向量回归 |
| `GLM` | 广义线性模型 |
| `TabPFN` | 基于预训练 Transformer 的表格模型 |

### 时间序列（2 个）

| 模型名 | 全称 |
|--------|------|
| `ARIMA` | 自回归差分滑动平均（含外生变量支持） |
| `ETS` | 指数平滑（趋势 + 季节） |

---

## 扩展指南

### 添加新模型（三步）

1. 选择 Mixin：`TorchMixin` / `SklearnMixin` / `TSMixin`
2. 实现对应的方法
3. 加一行 `@register` 注册

**示例 — 添加一个 sklearn 模型：**

```python
from EpiAI.models.base import SklearnMixin
from EpiAI.models.registry import register
import numpy as np

@register("MyModel", "my_model")
class MyForecaster(SklearnMixin):
    @classmethod
    def paradigm(cls):
        return "sklearn"

    def __init__(self, input_dim, lookback, horizon, target_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim
        self.model_ = MySklearnModel(**kwargs)

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        x_flat = self._flatten_x(train_x)
        y_prep = self._prepare_y(train_y)
        self.model_.fit(x_flat, y_prep)

    def predict(self, x):
        x_flat = self._flatten_x(x)
        raw = self.model_.predict(x_flat)
        return self._reshape_pred(raw)
```

保存文件到 `src/EpiAI/models/sklearn_models/my_model.py`，框架自动发现。
