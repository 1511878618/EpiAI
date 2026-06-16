# DeploymentRuntime API

## 初始化

```python
from EpiAI import ModelVault, DeploymentRuntime

vault = ModelVault.load("/path/to/vault/")
runtime = DeploymentRuntime(vault, time_col="time", time_unit="MS")
runtime.data_table = pd.read_csv("/path/to/history.csv")
```

**参数**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `vault` | `ModelVault` | — | `ModelVault.load()` 加载的模型库 |
| `time_col` | `str` | `"time"` | 数据中的时间列名 |
| `time_unit` | `str` | `"MS"` | 时间单位：`MS`(月) / `D`(日) / `h`(时) |
| `strict` | `bool` | `True` | `True`=时间不连续报错, `False`=仅警告 |

**模型来源**：训练好的模型文件放在 vault 目录中：

```
vault/
├── manifest.json
├── RF/
│   └── model.zip
└── ETS/
    └── model.zip
```

**历史数据**：`runtime.data_table` 是全部历史数据的 DataFrame，格式：

```
time       cases
2024-01    500
2024-02    300
...        ...
2026-03    1200
```

---

## feed()

每月调用一次，传入当月数据，返回全部模型的预测。

```python
new_data = pd.DataFrame({
    "time": ["2026-04-01"],
    "cases": [1500],
})
result = runtime.feed(new_data)
```

**输入**

| 列 | 类型 | 说明 |
|----|------|------|
| `time` | `str` | `"YYYY-MM-DD"` 格式，必须连续 |
| 其他列 | 同训练数据 | 模型需要的特征/目标列 |

时间连续性规则：

| 情况 | 行为 |
|------|------|
| 上月 `2026-03`，传入 `2026-04` | ✅ 正常 |
| 上月 `2026-03`，传入 `2026-05` | ❌ 抛 `TimeGapError` |
| 已传 `2026-04`，再次传入 `2026-04` | ❌ 抛 `TimeOrderError` |

**输出**

```python
{
    "RF": {
        "time":  ["2026-05", "2026-06", "2026-07"],   # 未来 3 个月
        "pred": [      1043,       1144,       1376],   # 预测值
    },
    "ETS": {
        "time":  ["2026-05", "2026-06", "2026-07"],
        "pred": [      1106,       1365,       1016],
    },
    "ARIMA": {
        "error": "..."   # 模型出错时有 error 字段
    },
}
```

| 字段 | 说明 |
|------|------|
| `time` | 预测对应的时间，`["YYYY-MM", ...]` |
| `pred` | 预测值数组，`pred[0]`=下月，`pred[1]`=下下月 |
| `error` | 模型出错时的错误信息 |

---

## 完整流程

```python
# 1. 加载模型（部署时做一次）
vault = ModelVault.load("/path/to/vault/")
runtime = DeploymentRuntime(vault, time_col="time")
runtime.data_table = pd.read_csv("/path/to/history.csv")

# 2. 每月收到新数据后执行
def on_new_data(time_str: str, cases: float):
    row = pd.DataFrame({"time": [time_str], "cases": [cases]})
    result = runtime.feed(row)

    for name, r in result.items():
        if "error" not in r:
            print(f"{name}: next month = {r['pred'][0]:.0f}")
    return result
```

---

## save()

```python
runtime.save("/path/to/runtime/")
```

输出目录：

```
runtime/
├── runtime_meta.json
├── data_table.parquet    # 或 data_table.csv（无 pyarrow 时）
└── vault/
```
