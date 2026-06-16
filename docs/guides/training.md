# 训练模块指南

## 功能介绍

训练模块位于 `EpiAI.trainer`，提供唯一的 `EpiAITrainer` 入口。

### EpiAITrainer

根据模型 `paradigm` 自动路由到对应的训练路径。

```python
from EpiAI.trainer import EpiAITrainer

# 三族模型使用完全相同的调用方式
result = EpiAITrainer(model=model).fit(bundle)

# Torch 可附加损失函数和优化器参数
result = EpiAITrainer(
    model=model,
    loss=nn.MSELoss(),
    optimizer_config={"lr": 1e-3, "max_epochs": 50},
    early_stopping_config={"patience": 10},
).fit(bundle)

# Sklearn 和 TS 模型自动忽略 loss / optimizer 参数
```

### 训练路径

| 族 | 内部方法 | 过程 |
|----|---------|------|
| torch | `_fit_torch()` | 设备自检 → AdamW → epoch 循环 → EarlyStopping |
| sklearn | `_fit_sklearn()` | 一次 `model.fit(train_x, train_y, val_x, val_y)` |
| ts | `_fit_ts()` | `fit_sequence(y_train)` → `predict_sequence(y_test)` |

### TrainResult

`fit()` 返回 `TrainResult`，包含预测结果和指标。

```python
result.predictions    # (N, horizon, target_dim)  已反标准化
result.metrics        # pd.DataFrame: MAE / RMSE / MAPE / R² / PearsonR
result.model          # 训练后的模型
result.history        # torch 的训练曲线（可选）
```

### 指标计算规则

- 只比 `y_pred[:, 0, :]`（第一步行预测），不混拼 horizon 步
- 窗口模型 y_true = `bundle.test_y[:, 0, :]`
- TS 模型 y_true = `get_y_series("test")[:n]`
- 预测结果自动做逆变换（反标准化）

---

## 扩展指南

### 添加新训练范式

如果未来引入新的模型类型（如 prophet、GluonTS），需要：

1. 在 `models/base.py` 定义新 Mixin：

```python
class ProphetMixin(BaseForecaster):
    @classmethod
    def paradigm(cls) -> str:
        return "prophet"

    def fit_sequence(self, y_train, X_train=None): ...
    def forecast(self, n_periods, X_future=None): ...
```

2. 在 `EpiAITrainer.fit()` 中添加路由：

```python
def fit(self, bundle):
    p = self.model.paradigm()
    if p == "torch":       return self._fit_torch(bundle)
    elif p == "sklearn":   return self._fit_sklearn(bundle)
    elif p == "ts":        return self._fit_ts(bundle)
    elif p == "prophet":   return self._fit_prophet(bundle)   # ← 新增
```

3. 实现 `_fit_prophet()`：

```python
def _fit_prophet(self, bundle):
    y_train = bundle.get_y_series("train").ravel()
    y_test = bundle.get_y_series("test").ravel()
    self.model.fit_sequence(y_train)
    preds = self.model.forecast(len(y_test))
    return self._build_result(bundle, preds)
```

### 自定义指标

`_compute_metrics()` 是模块级函数，可直接替换：

```python
from EpiAI.trainer import _compute_metrics

def my_metrics(y_true, y_pred, target_names):
    # 自定义指标逻辑
    ...

# 替换 trainer 中的函数引用
```
