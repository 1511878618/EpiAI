# DeploymentRuntime API

## 初始化

```python
from EpiAI.inference import DeploymentRuntime, ModelVault

vault = ModelVault.load("/path/to/vault/")
runtime = DeploymentRuntime(
    vault=vault,
    time_col="time",
    time_unit="MS",
    strict=True,
    data_table=history_df,       # 可选：直接传入历史数据
)
```

**参数**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `vault` | `ModelVault` | — | 已训练的模型集合 |
| `time_col` | `str` | `"time"` | 数据中的时间列名 |
| `time_unit` | `str` | `"MS"` | 时间单位：`MS`(月) / `D`(日) |
| `strict` | `bool` | `True` | 时间不连续时是否报错 |
| `data_table` | `pd.DataFrame` | `None` | 初始历史数据表（原始尺度） |

**历史数据格式**（`data_table` 必须存原始尺度）：

```csv
time,cases
2024-01,500
2024-02,300
...
2026-03,1200
```

---

## predict(horizon=1) → pd.DataFrame

预测未来 horizon 步，返回 时间×模型 表格。

```python
# 默认预测下 1 个月
result = runtime.predict()
print(result)
#            RF    ARIMA  Prophet
# 2026-05  1043    1106     989

# 预测 12 个月
result = runtime.predict(horizon=12)
print(result)
#            RF    ARIMA  Prophet
# 2026-05  1043    1106     989
# 2026-06  1144    1365    1012
# ...      ...      ...     ...
# 2027-04  1376    1016    1120
```

**各模型类型的处理方式：**

| 模型类型 | 预测逻辑 |
|---------|---------|
| **窗口模型** (RF/XGB/LSTM…) | `data_table` 最后 L 行 → predict → horizon 步 |
| **TS 模型** (ARIMA/Prophet/ETS…) | 从 `_last_ds` → `forecast(horizon)` → 输出 |

**TS 模型的 gap 补齐**：`_last_ds`（训练结束日期）可能早于 data_table 末尾。predict 自动计算 gap 并 `forecast(gap + horizon)` → 取最后 horizon 步，保证所有模型输出对齐。

---

## feed(new_data)

追加新观测数据到 data_table（不触发预测）。

```python
new_data = pd.DataFrame({
    "time": ["2026-04-01"],
    "cases": [1500],
})
runtime.feed(new_data)
```

**时间连续性规则**：

| 情况 | strict=True | strict=False |
|------|------------|-------------|
| 上月 `2026-03`，传入 `2026-04` | ✅ 正常 | ✅ 正常 |
| 上月 `2026-03`，传入 `2026-05` | ❌ `TimeGapError` | ⚠ 警告 |
| 已传 `2026-04`，再次传入 `2026-04` | ❌ `TimeOrderError` | ⚠ 警告 |

---

## update_ts(names=None)

用 data_table 中最近的数据重新拟合 TS 模型（滑动窗口）。

保持与原始训练相同的数据量，只取 data_table 中最近的 N 行重新拟合。
适用于 feed 新数据后让 TS 模型状态追赶上最新。

```python
runtime.feed(new_month_data)
runtime.update_ts()                    # TS 模型追到最新
preds = runtime.predict(horizon=12)    # 只 forecast 12 步
```

---

## retrain_all()

用 data_table 全部数据重新训练所有模型（窗口 + TS）。

需要先设置 `model_config`（记录各模型初始参数）。适用于部署初始化时用全量数据提升效果。

---

## save(path)

持久化运行时状态到磁盘。

```python
runtime.save("/path/to/runtime/")
```

输出目录：

```
runtime/
├── runtime_meta.json       # time_col, history_end_time, ...
├── data_table.parquet      # 或 data_table.csv（无 pyarrow 时）
└── vault/                  # 模型备份
```

## load(path)

从磁盘恢复运行时状态。

```python
runtime = DeploymentRuntime.load("/path/to/runtime/")
```

---

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `vault` | `ModelVault` | 当前模型集合 |
| `data_table` | `pd.DataFrame` | 历史观测数据（原始尺度） |
| `time_col` | `str` | 时间列名 |
| `_history_end_time` | `pd.Timestamp` | 历史数据末尾时间（自动推断） |

---

## 完整流程

```python
from EpiAI.inference import DeploymentRuntime, ModelVault

# 1. 加载模型与历史数据
vault = ModelVault.load("/path/to/vault/")
history = pd.read_csv("/path/to/history.csv")
runtime = DeploymentRuntime(vault=vault, data_table=history)

# 2. 预测未来
preds = runtime.predict(horizon=12)

# 3. 新数据到达
runtime.feed(pd.DataFrame({"time": ["2026-05-01"], "cases": [1600]}))
runtime.update_ts()                # TS 模型追到最新
preds = runtime.predict(horizon=6) # 重新预测

# 4. 持久化
runtime.save("/path/to/runtime/")

# 5. 加载恢复
runtime2 = DeploymentRuntime.load("/path/to/runtime/")
```

---

## 风险预警模块

```python
from EpiAI.risk import RiskScorer, WarningRule

scorer = RiskScorer(method="quantile").fit(runtime.data_table)
risk_df = scorer.score_df(runtime.predict(horizon=12))

rule = WarningRule(ensemble="max")
report = rule.evaluate(risk_df)
print(rule.report(report))
```

详见 `EpiAI.risk` 模块文档。
