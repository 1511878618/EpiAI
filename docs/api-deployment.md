# DeploymentRuntime API 参考

> 生产部署运行时。维护统一历史数据表，每次新数据到达时自动检查时间连续性并运行所有模型。

---

## 构造函数

```python
DeploymentRuntime(
    vault: ModelVault,
    time_col: str = "time",
    time_unit: str = "MS",
    strict: bool = True,
)
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `vault` | — | 已训练的模型库（`ModelVault.load()` 加载） |
| `time_col` | `"time"` | 数据中时间列的名称 |
| `time_unit` | `"MS"` | 时间单位：`"MS"`（月）、`"D"`（日）、`"h"`（时） |
| `strict` | `True` | `True` 时时间不连续抛异常；`False` 时仅警告 |

---

## 启动配置

初始化后需要加载历史数据，窗口模型才能正常预测（需要 `lookback` 行历史）。

```python
runtime = DeploymentRuntime(vault, time_col="time", time_unit="MS")

# 载入训练数据作为历史
runtime.data_table = training_data.copy()

# 设置训练结束时间（首次 feed 会检查是否衔接）
runtime._train_end_time = training_data["time"].iloc[-1]
```

---

## feed() — 核心方法

追加新数据、检查连续性、运行所有模型、自动持久化。

```python
result = runtime.feed(new_data)
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `new_data` | `pd.DataFrame` | ✅ | 新到达的数据行，需包含 `time_col` 以及模型需要的特征/目标列 |

### 返回结构

```python
{
    "RF": {
        "time": DatetimeIndex(["2026-05-01", "2026-06-01", "2026-07-01"]),
        "pred": ndarray([1043, 1144, 1376]),     # shape (horizon,)
    },
    "ETS": {
        "time": ...,
        "pred": ...,
    },
    # 如果某个模型出错:
    "ARIMA": {
        "error": "Model was fitted with X, so X_future must be provided.",
    },
}
```

### 内部流程

1. 检查时间连续性（缺口 → `TimeGapError`，乱序 → `TimeOrderError`）
2. 追加到 `data_table`
3. 遍历 vault 中的每个模型：
   - **窗口模型（torch / sklearn）**：从 `data_table.tail(lookback)` 取特征列 → `predict()`
   - **TS 模型（ARIMA / ETS）**：`forecast(horizon + feed_count)[-horizon:]` — 预测窗口随时间滑动
4. 自动 `save()` 到磁盘

---

## update_model() — 显式更新 TS 模型

更新单个 TS 模型的内部状态。与 `feed()` 独立，调用方在确认数据质量后执行。

```python
runtime.update_model("ETS", np.array([1200, 800]))
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 模型名（必须在 vault 中） |
| `new_y` | `np.ndarray` | 新观测值，形状 `(T,)` |

更新前自动备份当前 `y_history_` 到 `ts_backup/` 目录，可用于回滚。

---

## update_all_ts() — 批量更新

```python
runtime.update_all_ts(data: pd.DataFrame)
```

更新 vault 中所有 TS 模型。预留接口，后续可扩展为自动重训/增量学习。

---

## save() — 持久化

保存全部状态到磁盘。

```python
runtime.save("/path/to/runtime/")
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` 或 `Path` | 输出目录（自动创建） |

| 返回 | 说明 |
|------|------|
| `str` | 绝对路径 |

### 磁盘结构

```
runtime/
├── runtime_meta.json       # feed_count, 最新时间, 配置
├── data_table.parquet      # 全部历史数据（若无 pyarrow 则存 CSV）
├── vault/                  # ModelVault 目录（含全部模型）
└── ts_backup/              # TS 模型状态快照
    ├── ETS/
    │   ├── y_history_1.npy
    │   ├── y_history_2.npy
    │   └── ...
    └── ARIMA/
        ├── y_history_1.npy
        └── ...
```

---

## load() — 恢复

```python
runtime = DeploymentRuntime.load("/path/to/runtime/")
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` 或 `Path` | `save()` 生成的目录 |

| 返回 | 说明 |
|------|------|
| `DeploymentRuntime` | 恢复后的运行时，data_table 和 vault 完整可用 |

---

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `.data_table` | `pd.DataFrame` | 全部历史数据，可读写 |
| `.vault` | `ModelVault` | 已加载的模型库 |
| `._feed_count` | `int` | 已 feed 的次数 |

---

## 异常

| 异常 | 说明 |
|------|------|
| `TimeGapError` | 新数据与表中最新数据之间存在时间缺口 |
| `TimeOrderError` | 新数据时间 ≤ 表中最新时间（乱序或重复） |
| `BufferError` | `data_table` 行数不足以满足某模型的 `lookback` |

---

## 完整使用示例

```python
from EpiAI import ModelVault, DeploymentRuntime
import pandas as pd

# 加载模型库
vault = ModelVault.load("/path/to/vault/")

# 初始化运行时
runtime = DeploymentRuntime(vault, time_col="time", time_unit="MS")

# 载入历史数据
runtime.data_table = pd.read_csv("/path/to/history.csv")
runtime._train_end_time = pd.to_datetime("2026-03-01")

# 每月调用一次
new_observation = pd.DataFrame({
    "time": ["2026-04-01"],
    "cases": [1500],
})

result = runtime.feed(new_observation)

# 取预测结果
for name, r in result.items():
    if "error" not in r:
        print(f"{name}: {r['time']} → {r['pred']}")
```
