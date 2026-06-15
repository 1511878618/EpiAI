# EpiAI 训练架构设计（v3 定稿）

---

## 1. 三大模型族

```
Torch 族        → nn.Module，梯度下降训练，GPU
Sklearn 族      → sklearn 兼容，一次 fit，CPU  
TimeSeries 族   → 统计/传统时序，rolling origin 评价
```

命名统一：

| 族 | Paradigm | 基类 Mixin | 示例 |
|-----|----------|-----------|------|
| Torch | `"torch"` | `TorchMixin` | CNN, LSTM, Transformer, TimesNet |
| Sklearn | `"sklearn"` | `SklearnMixin` | XGB, LGBM, RandomForest, SVR, LinearReg, TabPFN |
| TimeSeries | `"ts"` | `TSMixin` | ARIMA, ETS |

---

## 2. 整体分层

```
┌─────────────────────────────────────────────────────────┐
│                   用户代码                                │
│  ForecasterRegistry.get("LSTM")(...)                     │
│  trainer.fit(model, bundle)                              │
├─────────────────────────────────────────────────────────┤
│                    EpiAITrainer                          │
│                                                         │
│  paradigm == "torch" → TorchTrainer                      │
│     epoch 循环, GPU, ADAMW, EarlyStopping, LR scheduler  │
│                                                         │
│  paradigm == "sklearn" → model.fit(train_x, train_y,     │
│     val_x, val_y)                                        │
│                                                         │
│  paradigm == "ts" → model.fit_sequence(y_train, X_train) │
│     model.predict_sequence(y_test, X_test)               │
│                                                         │
| 结果 → 反标准化 + 统一指标 → TrainResult                 |
├─────────────────────────────────────────────────────────┤
│                    PipelineBundle                        │
│  train_x/y (3D 窗) + train_df (变换后原始序列)            │
│  get_y_series() / get_X_series()                        │
├─────────────────────────────────────────────────────────┤
│                    ForecastPipeline                      │
│  Load → Split → Transform → Window                      │
└─────────────────────────────────────────────────────────┘

**关键实现细节：**
- TS 族如果模型不支持外生变量（如 ETS），Trainer 自动降级为单变量
- SklearnMixin 提供 `_flatten_x()` / `_prepare_y()` / `_reshape_pred()` 辅助
- Torch 模型缺少 torch 时可用 MockModule 占位，注册不受影响
```

---

## 3. BaseForecaster + Mixin

```python
class BaseForecaster(ABC):
    """所有预报模型的统一接口。"""

    @classmethod
    @abstractmethod
    def paradigm(cls) -> Literal["torch", "sklearn", "ts"]:
        ...

    # ── Torch / Sklearn 共用（窗口数据）────
    def fit(self, train_x, train_y, val_x=None, val_y=None):
        raise NotImplementedError

    def predict(self, x) -> np.ndarray:
        """返回 (N, horizon, target_dim)"""
        raise NotImplementedError

    # ── TimeSeries 专用（原始序列）────────
    def fit_sequence(self, y_train, X_train=None):
        raise NotImplementedError

    def predict_sequence(self, y_test, X_test=None,
                         update_state=True) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, n_periods, X_future=None) -> np.ndarray:
        raise NotImplementedError

    # ── 通用 ──────────────────────────────
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

### Mixin 定义

```python
class TorchMixin(BaseForecaster):
    """PyTorch 神经网络基类。扩展 nn.Module 使用。"""
    @classmethod
    def paradigm(cls) -> str:
        return "torch"

class SklearnMixin(BaseForecaster):
    """sklearn-like 模型。fit(x,y)/predict(x)。"""
    @classmethod
    def paradigm(cls) -> str:
        return "sklearn"
    # 提供 _flatten_x / _prepare_y / _reshape_pred 辅助

class TSMixin(BaseForecaster):
    """传统时序模型。fit_sequence(y)/predict_sequence(y)。"""
    @classmethod
    def paradigm(cls) -> str:
        return "ts"
```

---

## 4. 三类模型的具体形态

```python
# ── Torch ─────────────────────────────────────────────────
@register("LSTM", "lstm")
class LSTMForecaster(nn.Module, TorchMixin):
    def __init__(self, input_dim, lookback, horizon, target_dim, ...):
        super().__init__()
        ...

    def forward(self, x: Tensor) -> Tensor:
        return ...  # (B, H, T)

    # fit() 不实现，由 EpiAITrainer 的 TorchTrainer 接管
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(torch.tensor(x)).numpy()


# ── Sklearn ───────────────────────────────────────────────
@register("XGB", "XGBoost")
class XGBSingleForecaster(SklearnMixin):
    def __init__(self, input_dim, lookback, horizon, target_dim, ...):
        ...

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        x_flat = self._flatten_x(train_x)
        y_prep = self._prepare_y(train_y)
        if val_x is not None and hasattr(self.model_, 'set_params'):
            self.model_.set_params(eval_set=[(_flatten_x(val_x), _prepare_y(val_y))])
        self.model_.fit(x_flat, y_prep)

    def predict(self, x):
        return self._reshape_pred(self.model_.predict(self._flatten_x(x)))


# ── TimeSeries ────────────────────────────────────────────
@register("ARIMA", "auto_arima")
class AutoARIMAXRollingForecaster(TSMixin):
    def __init__(self, seasonal=True, m=12, ...):
        ...

    def fit_sequence(self, y_train, X_train=None):
        # y_train: (T,) — 原始 1D 时序
        self.model_ = auto_arima(y_train, X=X_train, ...)

    def predict_sequence(self, y_test, X_test=None, update_state=True):
        # Rolling origin: 逐点预测 → 更新状态
        preds = []
        for t in range(len(y_test)):
            pred = self.model_.predict(n_periods=1,
                                       X=X_test[t:t+1] if X_test is not None else None)
            preds.append(pred[0])
            if update_state:
                self.model_.update(y_test[t])
        return np.array(preds).reshape(-1, 1, 1)
```

---

## 5. 注册系统

```python
# src/EpiAI/models/registry.py

_registry: dict[str, type[BaseForecaster]] = {}

def register(*names: str):
    def wrapper(cls):
        for name in names:
            _registry[name.lower()] = cls
        return cls
    return wrapper

def get(name: str) -> type[BaseForecaster]:
    cls = _registry.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown: {name}. Available: {list(_registry.keys())}")
    return cls

def list_models(paradigm=None) -> list[str]:
    if paradigm is None:
        return sorted(_registry.keys())
    return sorted(k for k, v in _registry.items() if v.paradigm() == paradigm)
```

---

## 6. PipelineBundle 变更

```python
@dataclass
class PipelineBundle:
    # ...现有字段不变...
    train_df: Optional[pd.DataFrame] = None  # 新增
    val_df:   Optional[pd.DataFrame] = None  # 新增
    test_df:  Optional[pd.DataFrame] = None  # 新增

    def get_y_series(self, split="train") -> np.ndarray:
        """返回 (T, target_dim)。给 TS 族用。"""
        df = getattr(self, f"{split}_df")
        return df[self.target_names].values.astype(np.float32)

    def get_X_series(self, split="train") -> np.ndarray:
        """返回 (T, n_features)。"""
        df = getattr(self, f"{split}_df")
        return df[self.feature_names].values.astype(np.float32)
```

`ForecastPipeline.run()` 新增三行：

```python
return PipelineBundle(..., train_df=train_df, val_df=val_df, test_df=test_df)
```

---

## 7. EpiAITrainer — 统一训练入口

```python
class EpiAITrainer:
    """统一训练器。根据 paradigm 自动路由。
    
    参数
    ----
    model : BaseForecaster
        已实例化的模型。
    loss : nn.Module or None
        Torch 族的损失函数。Sklearn / TS 自动忽略。
    optimizer_config : dict or None
        Torch 族的优化器参数。其余忽略。
    early_stopping_config : dict or None
        Torch 族的早停参数。其余忽略。
    device : str
        Torch 族的设备。
    verbose : bool
    """

    def __init__(self, model: BaseForecaster,
                 loss=None,
                 optimizer_config=None,
                 early_stopping_config=None,
                 device="auto",
                 verbose=True):
        self.model = model
        self.loss = loss
        self.optimizer_config = optimizer_config or {}
        self.early_stopping_config = early_stopping_config or {}
        self.device = device
        self.verbose = verbose

    def fit(self, bundle: PipelineBundle) -> TrainResult:
        p = self.model.paradigm()

        if p == "torch":
            return self._fit_torch(bundle)
        elif p == "sklearn":
            return self._fit_sklearn(bundle)
        elif p == "ts":
            return self._fit_ts(bundle)
        else:
            raise ValueError(f"Unknown paradigm: {p}")

    # ── Torch ──────────────────────────────────────────────
    def _fit_torch(self, bundle):
        device = ...  # 自检 cuda / mps / cpu
        model = self.model.to(device)
        loss_fn = self.loss or nn.MSELoss()
        optimizer = AdamW(model.parameters(), **self.optimizer_config)
        early_stopper = EarlyStopping(**self.early_stopping_config)

        train_loader = DataLoader(TensorDataset(bundle.train_x, bundle.train_y),
                                  batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(bundle.val_x, bundle.val_y),
                                batch_size=32)

        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
            early_stopper.step(val_loss, model)
            if early_stopper.should_stop:
                break

        preds = self.model.predict(bundle.test_x)
        return self._build_result(bundle, preds)

    # ── Sklearn ────────────────────────────────────────────
    def _fit_sklearn(self, bundle):
        self.model.fit(
            bundle.train_x, bundle.train_y,
            val_x=bundle.val_x, val_y=bundle.val_y,
        )
        preds = self.model.predict(bundle.test_x)
        return self._build_result(bundle, preds)

    # ── TimeSeries ─────────────────────────────────────────
    def _fit_ts(self, bundle):
        y_train = bundle.get_y_series("train").squeeze()
        y_test = bundle.get_y_series("test").squeeze()

        # Try with X features first, fall back to univariate
        try:
            X_train = bundle.get_X_series("train") if bundle.feature_names else None
            X_test = bundle.get_X_series("test") if bundle.feature_names else None
            self.model.fit_sequence(y_train, X_train)
            preds = self.model.predict_sequence(y_test, X_test, update_state=True)
        except (ValueError, TypeError) as e:
            if "X" in str(e) or "exogenous" in str(e):
                self.model.fit_sequence(y_train, None)
                preds = self.model.predict_sequence(y_test, None, update_state=True)
            else:
                raise
        return self._build_result(bundle, preds)

    # ── 统一结果 ───────────────────────────────────────────
    def _build_result(self, bundle, predictions) -> TrainResult:
        if bundle.transforms is not None:
            predictions = self._inverse_target(predictions, bundle)
        y_true = bundle.get_y_series("test")
        metrics = _compute_metrics(y_true, predictions, bundle.target_names)
        return TrainResult(model=self.model, predictions=predictions,
                           metrics=metrics, history=getattr(self, "_history", None))
```

---

## 8. TrainResult

```python
@dataclass
class TrainResult:
    model: BaseForecaster        # 训练后的模型
    predictions: np.ndarray      # (N, horizon, target_dim) 已反标准化
    metrics: pd.DataFrame        # MAE/RMSE/MAPE/R²/PearsonR per target per split
    history: Optional[dict]      # torch 的训练曲线
```

---

## 9. 用户使用示例（三种模型完全相同）

```python
bundle = ForecastPipeline.quick(
    path="china_climate.csv",
    time_col="Year/Month", target_cols="登革热",
    feature_cols=["t2m_mean", "tcc_mean", "乙脑"],
    entity_col="province",
    lookback=12, horizon=3,
)

# ── Torch ──
model = ForecasterRegistry.get("LSTM")(
    input_dim=bundle.n_features, lookback=12, horizon=3, target_dim=1)
result = EpiAITrainer(
    model=model,
    loss=OutbreakAwareLoss(threshold=100),
    optimizer_config={"lr": 1e-3},
    early_stopping_config={"patience": 10},
).fit(bundle)

# ── Sklearn ──
model = ForecasterRegistry.get("XGB")(
    input_dim=bundle.n_features, lookback=12, horizon=3, target_dim=1)
result = EpiAITrainer(model=model).fit(bundle)
#         ↑ loss, optimizer, early_stopping 自动忽略

# ── TimeSeries ──
model = ForecasterRegistry.get("ARIMA")(seasonal=True, m=12)
result = EpiAITrainer(model=model).fit(bundle)

# 结果格式完全一样
print(result.metrics)
print(result.predictions.shape)  # (N_test, 3, 1)
```

---

## 10. 文件结构

```
src/EpiAI/
├── dataset/           ✅ 已完成
│   ├── pipeline.py    → PipelineBundle + get_y_series()
│   ├── base.py        → Transform / Compose / SplitStrategy ABC
│   ├── container.py   → TimeSeriesData / SplitResult / WindowBundle
│   ├── loaders.py     → CsvLoader / FeatherLoader / TensorLoader
│   ├── splits.py      → 6种拆分策略
│   ├── transforms.py  → 8种变换 + SlidingWindow
│   └── ...
├── models/
│   ├── __init__.py        导出 register / get / list_models
│   ├── base.py            BaseForecaster + TorchMixin/SklearnMixin/TSMixin
│   ├── registry.py        @register 注册系统
│   ├── torch_models/      10个模型 → @register + TorchMixin
│   ├── sklearn_models/    6个模型  → @register + SklearnMixin
│   └── ts_models/         2个模型  → @register + TSMixin
├── trainer.py             EpiAITrainer（三路路由）
├── losses.py              14个损失函数（保持不动）
│
# 旧代码保留（不维护，不移除）：
├── train.py               旧训练循环
└── time_serie_task.py     旧调度器
```

---

## 11. 设计原则检查

| 原则 | 体现 |
|------|------|
| **扩展性** | 加一个模型 = 一个 .py 文件 + 一行 `@register`，不改别处 |
| **闭合性** | 三种范式内部彻底隔离，Trainer 只有 3 路 dispatch |
| **便捷性** | `get("LSTM")` + `trainer.fit(bundle)` 两行搞定 |
| **运行效率** | Torch 走完整训练循环；Sklearn 一次 fit；TS 用 rolling origin |
| **loss/optim 零干扰** | Sklearn/TS 无视这些参数，调用方不需要操心 |

---

## 12. 测试覆盖

```
7 个测试文件，62 个测试用例，全部通过

Phase 1: 数据集抽象层   7 tests
Phase 2: DataLoaders    7 tests
Phase 3: SplitStrategy  12 tests
Phase 4: Transforms     15 tests
Phase 5: ForecastPipeline 8 tests
Phase 6: 集成验证       5 tests
Phase 7: 端到端集成      8 tests

注册模型总数: 29
  torch:   11 (CNN, LSTM, CNN-LSTM, MLP, ResNet, TCN,
             Transformer, DLinear, Autoformer, TimesNet)
  sklearn: 13 (XGB, LGBM, TabPFN, RF, SVR, LinearReg + 别名)
  ts:       5 (ARIMA, ETS + 别名)
```
