# EpiAI 训练架构设计（v4 定稿）

---

## 1. 整体分层

```
┌──────────────────────────────────────────────────────────────┐
│                         用户代码                              │
│  get("LSTM")(...).fit(bundle)  /  vault.best("R2")           │
│  runtime.feed(new_data)                                      │
├──────────────────────────────────────────────────────────────┤
│                     InferenceLayer                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  DeploymentRuntime  (生产部署)                          │    │
│  │  ┌─ data_table (持久化)                                │    │
│  │  ├─ feed() → 时间检查 → 追加 → 各模型推理 → 持久化      │    │
│  │  └─ update_model() / update_all_ts()                    │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │  ModelVault  (多模型管理)                               │    │
│  │  ├─ best() / summary() / predict_all()                 │    │
│  │  └─ save() / load() → zip 包                          │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │  InferencePipeline  (单模型推理)                        │    │
│  │  ├─ predict(df) → 变换 → 窗口 → 预测 → 反标准化       │    │
│  │  ├─ forecast(steps) / update(y)                       │    │
│  │  └─ save() / load()                                   │    │
│  └──────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────┤
│                     EpiAITrainer                              │
│                                                              │
│  paradigm == "torch"    → epoch 循环 + AdamW + EarlyStopping │
│  paradigm == "sklearn"  → model.fit(train_x, train_y, ...)   │
│  paradigm == "ts"       → fit_sequence(y) + predict_sequence │
│                                                              │
│  结果 → inverse_predictions() → _compute_metrics()          │
│       → TrainResult                                          │
├──────────────────────────────────────────────────────────────┤
│                     PipelineBundle                            │
│  train/val/test_x/y (3D 窗口) + train/val/test_df (原始序)   │
│  get_y_series() / get_X_series()                             │
│  val/test 窗口已带 lookback+horizon-1 行上下文，无数据丢失   │
├──────────────────────────────────────────────────────────────┤
│                     ForecastPipeline                          │
│  Load → Split → Transform → Window                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 三大模型族

| 族 | Paradigm | 训练方式 | 基类 | 示例模型 |
|----|----------|---------|------|---------|
| Torch | `"torch"` | 梯度下降，GPU | `TorchMixin` | MLP, LSTM, CNN, ResNet, TCN, Transformer, DLinear, Autoformer, TimesNet |
| Sklearn | `"sklearn"` | 一次 fit | `SklearnMixin` | RF, XGB, LGBM, SVR, GLM, TabPFN |
| TimeSeries | `"ts"` | 统计推断 | `TSMixin` | ARIMA, ETS |

三种模型训练接口统一：

```python
# Torch — 走完整训练循环
EpiAITrainer(model=get("LSTM")(...), loss=OutbreakAwareLoss(...)).fit(bundle)

# Sklearn — optimizer/loss 自动忽略
EpiAITrainer(model=get("XGB")(...)).fit(bundle)

# TS — 自动走 fit_sequence + rolling origin
EpiAITrainer(model=get("ETS")(seasonal_periods=12)).fit(bundle)
```

---

## 3. BaseForecaster + Mixin

```python
class BaseForecaster(ABC):
    @classmethod
    @abstractmethod
    def paradigm(cls) -> Literal["torch", "sklearn", "ts"]: ...

    # ── 窗口模型（Torch / Sklearn）────
    def fit(self, train_x, train_y, val_x=None, val_y=None): ...
    def predict(self, x) -> np.ndarray: ...   # (N, horizon, target_dim)

    # ── 时序模型（TS）───────────────
    def fit_sequence(self, y_train, X_train=None): ...
    def predict_sequence(self, y_test, X_test=None, update_state=True): ...
    def forecast(self, n_periods, X_future=None) -> np.ndarray: ...
```

**TorchMixin**：提供默认 `predict()` → `forward()` + `no_grad()` + 自动设备匹配（`.to(device)`）。

**SklearnMixin**：提供 `_flatten_x()` / `_prepare_y()` / `_reshape_pred()` 辅助方法。

**TSMixin**：提供 ABC，子类必须实现 `fit_sequence` / `predict_sequence` / `forecast`。

---

## 4. 注册系统

```python
# registry.py
_registry: dict[str, type[BaseForecaster]] = {}

def register(*names: str):
    def wrapper(cls):
        for name in names:
            _registry[name.lower()] = cls
        return cls
    return wrapper

def get(name: str) -> type[BaseForecaster]:
    return _registry[name.lower()]

def list_models(paradigm=None) -> list[str]:
    if paradigm is None:
        return sorted(_registry.keys())
    return sorted(k for k, v in _registry.items() if v.paradigm() == paradigm)
```

---

## 5. EpiAITrainer — 统一训练入口

### 路由逻辑

```python
def fit(self, bundle) -> TrainResult:
    p = self.model.paradigm()
    if p == "torch":    return self._fit_torch(bundle)
    elif p == "sklearn": return self._fit_sklearn(bundle)
    elif p == "ts":     return self._fit_ts(bundle)
```

### Torch 路径

- 设备自检（cuda / mps / cpu）
- 损失函数默认 `nn.MSELoss()`，可自定义
- AdamW 参数过滤：只传 `lr / weight_decay / betas / eps / amsgrad`，防止 `max_epochs` / `batch_size` 误传
- EarlyStopping 支持
- epoch 循环 + 验证

### Sklearn 路径

- 调用 `model.fit(train_x, train_y, val_x, val_y)`
- 各模型自行处理 val 参数（RF/XGB 用于 early stopping，SVR/GLM 忽略）

### TS 路径

- 剥离 X 中与 target 重叠的列（防止 ARIMA 看到未来真值作为外生变量）
- 调用 `fit_sequence(y, X)` → `predict_sequence(y, X)`
- 如果模型不支持外生变量（如 ETS），自动降级为单变量

### 结果组装

```python
def _build_result(self, bundle, predictions) -> TrainResult:
    paradigm = self.model.paradigm()
    if paradigm == "ts":
        n = predictions.shape[0]
        y_true = bundle.get_y_series("test")[:n]
    else:
        y_true = bundle.test_y[:, 0, :]   # 窗口目标，只取首步

    predictions = inverse_predictions(
        predictions, bundle.target_names, bundle.transforms, y_true=y_true,
    )
    metrics = _compute_metrics(y_true, predictions, bundle.target_names)
    return TrainResult(model, predictions, metrics, history)
```

### 共享模块级函数

```python
def inverse_predictions(predictions, target_names, transforms, y_true=None):
    """逐列逆运算，只操作 predictions[:, 0, :]（首步）。"""

def _compute_metrics(y_true, y_pred, target_names) -> pd.DataFrame:
    """只比 y_pred[:, 0, :]（首步）。指标：MAE / RMSE / MAPE / R² / PearsonR。"""
```

---

## 6. 窗口上下文 — 数据丢失修复

val/test 窗口创建时，自动拼接前一个 split 的尾部 `lookback + horizon - 1` 行作为上下文：

```
上下文 14 行 + test 30 行 = 44 行
窗口: 44 - 12 - 3 + 1 = 30 个
首步覆盖: test[0..29] = 全部 30 个测试点 ✅
```

之前只取 `lookback` 行上下文，导致首步只能覆盖 `test[0..27]`，最后 `horizon-1` 行丢失。

---

## 7. TrainResult

```python
@dataclass
class TrainResult:
    model: BaseForecaster
    predictions: np.ndarray      # (N, horizon, target_dim) 已反标准化
    metrics: pd.DataFrame        # MAE / RMSE / MAPE / R² / PearsonR
    history: Optional[dict]      # torch 的训练曲线
```

---

## 8. InferencePipeline — 单模型推理

```python
inferer = InferencePipeline.from_train_result(result)

# 窗口模型：输入 DataFrame → 变换 → 窗口 → 预测 → 反标准化
pred = inferer.predict(new_data_df)

# TS 模型：纯未来预测 / 在线更新
forecast = inferer.forecast(steps=6)
updated = inferer.update(new_observations)

# 持久化
inferer.save("model.zip")
inferer = InferencePipeline.load("model.zip")
```

---

## 9. ModelVault — 多模型管理

```python
vault = ModelVault.from_results({"RF": r_rf, "XGB": r_xgb}, bundle)
vault.save("/tmp/vault/")
vault.summary()                     # 对比表（按 R² 降序）
vault.best("R2")                    # 选最优
vault.predict_all(new_data)         # 批量推理
vault["RF"]                         # 获取单模型 InferencePipeline
vault = ModelVault.load("/tmp/vault/")
```

目录结构：

```
/tmp/vault/
├── manifest.json           # 全部模型及指标
├── RF/
│   ├── model.zip           # InferencePipeline 包
│   └── meta.json           # 训练参数
└── ETS/
    ├── model.zip
    └── meta.json
```

---

## 10. DeploymentRuntime — 生产部署

### 核心概念

- **统一 data_table**：持久化 DataFrame（parquet/CSV），包含全部历史数据
- **feed()**：追加新数据 → 时间连续性检查 → 各模型按需取数据推理 → 自动持久化
- **TS 模型不自动 update**：显式调用 `update_model()` 更新状态

### feed 流程

```python
def feed(self, new_data):
    _check_time_continuity(new_data)
    self.data_table.concat(new_data)

    for name, inferer in self.vault.models.items():
        if inferer.paradigm == "ts":
            # 滑动预测：forecast(horizon + feed_count)[-horizon:]
            n = horizon + max(0, feed_count - 1)
            raw = inferer.forecast(n)[-horizon:]
        else:
            # 窗口模型：取 data_table.tail(lookback) → predict
            x = self.data_table.tail(lookback)[feature_cols]
            raw = inferer.predict(x)

    self._persist()
```

### 时间连续性检查

| 场景 | 行为 |
|------|------|
| 新数据紧接表中最后一条 | ✅ 正常追加 |
| 存在时间缺口 | ❌ `TimeGapError` |
| 新数据 ≤ 表中最后时间 | ❌ `TimeOrderError` |
| 首次 feed 不接训练结束时间 | ❌ `TimeGapError` |
| strict=False | ⚠️ 仅警告 |

### TS 状态管理

```python
# 显式更新（需确认数据质量）
runtime.update_model("ETS", np.array([1200, 800]))

# 批量更新（预留接口，未来可扩展为自动重训）
runtime.update_all_ts(new_df)
```

---

## 11. 数据契约一致性（已修复的 Bug）

| 问题 | 修复 | 涉及函数 |
|------|------|---------|
| 列的 alignment | `self.mean_[cols].values` 过滤 | StandardScaler.transform/inverse, RobustScaler |
| 窗口 y_true 不对齐 | 窗口用 `test_y[:,0,:]`，TS 用 `get_y_series` | `_build_result` |
| horizon 混比 | 只用 `y_pred[:, 0, i]`（首步） | `_compute_metrics`, `inverse_predictions` |
| 测试集未覆盖尾行 | 上下文 = `lookback + horizon - 1` | pipeline.run() 窗口逻辑 |
| 推理时缺少目标列 | `SlidingWindow.apply_features_only()` | InferencePipeline.predict() |
| ARIMA 外生变量泄漏 | 剥离 X 中与 target 重叠的列 | `_fit_ts` |
| AdamW kwargs | 只传 lr/weight_decay 等有效参数 | `_fit_torch` |
| Torch device 不匹配 | `x_t.to(next(self.parameters()).device)` | TorchMixin.predict() |

---

## 12. 文件结构

```
src/EpiAI/
├── dataset/
│   ├── base.py        ← Transform / SplitStrategy / Compose ABC
│   ├── container.py   ← TimeSeriesData / SplitResult / WindowBundle
│   ├── loaders.py     ← CsvLoader / FeatherLoader / TensorLoader
│   ├── splits.py      ← 6 种拆分策略
│   ├── transforms.py  ← 8 种变换 + SlidingWindow
│   └── pipeline.py    ← ForecastPipeline → PipelineBundle
├── models/
│   ├── __init__.py    ← 导出 register / get / list_models
│   ├── base.py        ← BaseForecaster + 3 Mixin
│   ├── registry.py    ← @register 注册系统
│   ├── torch_models/  ← 10 个模型（@register + TorchMixin）
│   ├── sklearn_models/ ← 6 个模型（@register + SklearnMixin）
│   └── ts_models/     ← 2 个模型（@register + TSMixin）
├── trainer.py         ← EpiAITrainer + inverse_predictions + _compute_metrics
├── inference.py       ← InferencePipeline + ModelVault + DeploymentRuntime
├── losses.py          ← 14 个损失函数（保持不动）
├── train.py           ← 旧训练循环（保持不动）
└── time_serie_task.py ← 旧调度器（保持不动）
```

---

## 13. 设计原则

| 原则 | 体现 |
|------|------|
| **扩展性** | 加一个模型 = 一个 .py + 一行 `@register`，不改别处 |
| **闭合性** | 三种范式内部分离，Trainer 只有 3 路 dispatch |
| **数据契约** | 所有修复都围绕「列、行、时间步」三维度做一致性维护 |
| **部署就绪** | 从 `InferencePipeline` 到 `ModelVault` 到 `DeploymentRuntime`，逐层递进 |
