# DeploymentRuntime — 生产部署运行时

`DeploymentRuntime` 是 EpiAI 的生产部署组件，管理已训练模型的推理、数据更新。

---

## 设计理念

```
训练期 (全部已观测)             部署期
┌──────────┬────────┬────────┐ ┌──────┬──────┬──────┐
│  train   │  val   │  test  │ │feed1 │feed2 │ ...  │ 时间→
└──────────┴────────┴────────┘ └──────┴──────┴──────┘
▲data_table 末尾                ↑predict(horizon=12) 输出
```

- **data_table**：存储全部历史观测数据，**原始尺度**（与训练时 CSV 一致）
- **Transform 透明**：`InferencePipeline` 内部自动 `apply_transforms` → 模型推理 → `inverse_target`，用户只接触原始值
- **TS 模型 gap 补齐**：`_last_ds` 到 data_table 末尾的 gap 通过 `forecast(gap+horizon)` 跳过，输出对齐

---

## API 参考

### `__init__(vault, time_col, time_unit, strict, data_table)`

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `vault` | `ModelVault` | — | `ModelVault.load()` 或 `ModelVault.from_results()` |
| `time_col` | `str` | `"time"` | 时间列名 |
| `time_unit` | `str` | `"MS"` | `"MS"`(月), `"D"`(日) |
| `strict` | `bool` | `True` | 时间不连续报错 |
| `data_table` | `pd.DataFrame` | `None` | 初始历史数据（原始尺度） |

`_history_end_time` 自动从 `data_table` 末尾推断。

---

### `predict(horizon=1) → pd.DataFrame`

预测未来 horizon 步，返回 时间×模型 表格。

```python
# 默认：预测下 1 个月
runtime.predict()
# 输出:
#            RF   ARIMA  Prophet  ...
# 2026-05  1043    1106     989

# 预测 12 个月
runtime.predict(horizon=12)
```

| 模型类型 | 实现 |
|---------|------|
| 窗口模型 | `data_table` 最后 L 行 → `predict` → H 步 |
| TS 模型 | `forecast(gap + horizon)` → 取最后 horizon 步 |

TS 模型的 gap 补齐：`forecast(gap + horizon) -> pred[-horizon:]`，保证所有模型时间对齐。

---

### `feed(new_data)`

追加新观测到 data_table（不触发预测，不复位模型）。

```python
runtime.feed(pd.DataFrame({"time": ["2026-05-01"], "cases": [1500]}))
```

---

### `update_ts(names=None)`

用 data_table 中最近的数据重新拟合 TS 模型（滑动窗口）。

保持原始训练数据量，窗口滑动到最新。适合 feed 后让 TS 模型赶上最新状态。

```python
runtime.feed(new_month)
runtime.update_ts()
runtime.predict(horizon=12)   # 此时 TS 模型从 data_table 末尾 forecast
```

内部流程：
1. 从模型读取 `train_size_` / `_y_train` 确定窗口大小
2. 从 data_table 取最近 N 行
3. 通过 `inferer.transforms.transform()` 变换到模型空间
4. 调用 `model.fit_sequence()` 重新拟合
5. 更新 `_last_ds`

---

### `save(path)` / `load(path)`

持久化/加载运行时状态。

```
runtime/
├── runtime_meta.json
├── data_table.parquet
└── vault/
```

---

## 完整示例

```python
from EpiAI.inference import ModelVault, DeploymentRuntime
from EpiAI.risk import RiskScorer, WarningRule
import pandas as pd

# 加载
vault = ModelVault.load("/path/to/vault/")
history = pd.read_csv("/path/to/history.csv")
runtime = DeploymentRuntime(vault=vault, data_table=history)

# 预测
preds = runtime.predict(horizon=12)
print(preds)

# feed + update
runtime.feed(pd.DataFrame({"time": ["2026-05-01"], "cases": [1600]}))
runtime.update_ts()
preds = runtime.predict(horizon=6)

# 风险预警
scorer = RiskScorer(method="quantile").fit(runtime.data_table)
risk_df = scorer.score_df(preds)
rule = WarningRule(ensemble="max")
report = rule.evaluate(risk_df)
print(rule.report(report))

# 持久化
runtime.save("/tmp/my_runtime/")
```

---

## 注意

| 事项 | 说明 |
|------|------|
| **data_table 必须存原始值** | 模型内部 `InferencePipeline` 自动处理 transform/inverse |
| **TS 模型需要 update_ts** | 否则 gap 过长会导致 forecast 收敛到均值 |
| **predict() 返回 DataFrame** | 不再是 dict，直接可用 `pandas` 操作 |
