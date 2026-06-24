# DeploymentRuntime — 生产部署运行时

`DeploymentRuntime` 是 EpiAI 的生产部署组件，管理已训练模型的推理、数据更新和重新训练。

---

## 设计理念

```
训练期 (全部已观测)             部署期
┌──────────┬────────┬────────┐ ┌──────┬──────┬──────┐
│  train   │  val   │  test  │ │feed1 │feed2 │ ...  │ 时间→
└──────────┴────────┴────────┘ └──────┴──────┴──────┘
▲_train_end_time                ↑predict(目标时间点)
```

- **data_table**：维护全部历史观测数据（训练集 + 验证集 + 测试集 + 部署后新数据）
- **vault**：包含所有已训练模型（窗口模型 + TS 模型）
- **predict(target_time)**：对任意未来时间点统一查询各模型预测结果

---

## 核心 API

### `__init__(vault, time_col, time_unit, strict)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `vault` | `ModelVault` | 已训练的模型集合 |
| `time_col` | `str` | 数据表中时间列的名称（默认 `"time"`） |
| `time_unit` | `str` | pandas 时间频率别名（`"MS"` 月, `"D"` 日, 默认 `"MS"`） |
| `strict` | `bool` | 是否严格检查时间连续性（默认 `True`） |

### `predict(target_time) -> Dict[str, float | dict]`

预测指定时间点的值。

| 参数 | 类型 | 说明 |
|------|------|------|
| `target_time` | `str / datetime / pd.Timestamp` | 要预测的时间点 |

返回 `{模型名: 预测值}`，预测失败时值为 `{"error": "..."}`。

**各模型类型的处理方式：**

| 模型类型 | 预测逻辑 |
|---------|---------|
| **窗口模型** (RF/XGB/LSTM/CNN…) | 取 `data_table` 最后 L 行做 lookback → `model.predict(lookback)` |
| **TS 模型** (ARIMA/Prophet/ETS…) | 从 `model._last_ds` forecast 到 `target_time` → 取该步的值 |

> 窗口模型预测未来时，`target_time` 不需要存在于 `data_table` 中。
> 引擎会自动使用最近 L 行观测数据作为输入。

### `predict_range(start_time, end_time) -> Dict[str, np.ndarray]`

批量预测时间范围内的每个时间点。

| 参数 | 类型 | 说明 |
|------|------|------|
| `start_time` | `str / datetime` | 起始时间 |
| `end_time` | `str / datetime` | 结束时间 |

返回 `{模型名: np.array([预测值, ...])}`。

### `feed(new_data)`

追加新的观测数据到 data_table（不触发预测）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `new_data` | `pd.DataFrame` | 包含 `time_col` 列和 feature/target 列的新数据 |

- 自动检查时间连续性（按 `time_unit` 步长）
- `strict=True` 时，时间不连续或乱序会抛出 `TimeGapError` / `TimeOrderError`
- `strict=False` 时仅发出警告

### `retrain_all()`

用 data_table 中的全部数据重新训练所有模型并更新 vault。

**前置条件**：每个 `InferencePipeline.model_config` 必须已正确设置：

```python
for name, inferer in vault.models.items():
    for disp, mname, kwargs in MODEL_DEFS:
        if disp == name:
            inferer.model_config = {
                "register_name": mname,     # 注册名，用于 get(name)
                "init_kwargs": kwargs,       # 创建模型时的参数字典
            }
```

**各模型类型的重训逻辑：**

| 模型类型 | 重训流程 |
|---------|---------|
| **窗口模型** | 用全部数据重建 `ForecastPipeline`（`train_ratio=1.0`）→ `EpiAITrainer.fit(bundle)` |
| **TS 模型** | 取全量 y 序列 → `model.fit_sequence(y_full, dates=dates)` |

重训后自动更新 `_train_end_time` 为 data_table 最后时间点。

### `save(path)`

持久化运行时状态到磁盘。

```
path/
├── runtime_meta.json       # 元数据 (time_col, time_unit, train_end_time, …)
├── data_table.parquet      # 历史观测数据（parquet / CSV fallback）
└── vault/                  # 模型 vault（每模型子目录 + manifest.json）
```

### `load(path) -> DeploymentRuntime`

从磁盘恢复运行时状态。

---

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `vault` | `ModelVault` | 当前模型集合 |
| `data_table` | `pd.DataFrame` | 历史观测数据表 |
| `time_col` | `str` | 时间列名 |
| `_train_end_time` | `pd.Timestamp / None` | 训练结束时间（部署起点） |

---

## 快速开始

```python
from EpiAI.inference import DeploymentRuntime, ModelVault

# 1. 初始化 runtime，传入 vault 和全部历史数据
runtime = DeploymentRuntime(vault=vault, time_col="time", time_unit="MS")
runtime.data_table = history_df.copy()
runtime._train_end_time = pd.to_datetime(history_df["time"].iloc[-1])

# 2. 预测未来
result = runtime.predict("2026-05-01")
print(result["RF"])        # 窗口模型预测值
print(result["Prophet"])   # TS 模型预测值

# 3. 批量预测
results = runtime.predict_range("2026-05-01", "2026-10-01")

# 4. 新数据到达
runtime.feed(new_observations_df)

# 5. 用全部数据重训
for name, inferer in vault.models.items():
    for disp, mname, kwargs in MODEL_DEFS:
        if disp == name:
            inferer.model_config = {"register_name": mname, "init_kwargs": kwargs}
runtime.retrain_all()

# 6. 持久化
runtime.save("/path/to/runtime/")

# 7. 恢复
runtime = DeploymentRuntime.load("/path/to/runtime/")
```

---

## 错误类型

| 异常 | 触发条件 |
|------|---------|
| `TimeGapError` | feed 的数据与 data_table 之间存在时间跳跃 |
| `TimeOrderError` | feed 的数据时间戳 <= data_table 最后时间 |
| `BufferError` | data_table 行数不足窗口模型的 lookback |
