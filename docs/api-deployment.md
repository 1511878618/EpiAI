# DeploymentRuntime 使用说明

> 只需要知道四件事：模型从哪里来、历史数据怎么放、喂什么数据、吐出什么结果。

---

## 1. 模型从哪里来

训练好的模型存放在一个 **vault 目录**中，`DeploymentRuntime` 启动时加载它。

```python
from EpiAI import ModelVault, DeploymentRuntime

vault = ModelVault.load("/path/to/vault/")       # ← 别人训练好给你的
runtime = DeploymentRuntime(vault, time_col="time", time_unit="MS")
```

vault 目录结构：
```
vault/
├── manifest.json       # 有哪些模型、各自的指标
├── RF/
│   └── model.zip       # 单个模型
└── ETS/
    └── model.zip
```

---

## 2. 历史数据怎么存放

`DeploymentRuntime` 内部有一张 **data_table**，存全部历史数据。启动时从 CSV 加载：

```python
runtime.data_table = pd.read_csv("history.csv")   # ← 载入历史
```

history.csv 格式：
```csv
time,cases
2024-01,500
2024-02,300
...
2026-03,1200
```

`data_table` 会随着每次 feed 自动增长，无需手动管理。

---

## 3. 喂什么数据（feed 的输入）

每月调用一次 `feed()`，传入当月的新数据：

```python
new_row = pd.DataFrame({
    "time": ["2026-04-01"],
    "cases": [1500],
})

result = runtime.feed(new_row)
```

**输入格式要求：**

| 要求 | 说明 |
|------|------|
| 列名 | 必须和训练数据一致（如 `time`、`cases`） |
| `time` | 时间值，格式 `"YYYY-MM-DD"` |
| 其他列 | 模型需要的特征列（如 `cases`、`temp` 等） |
| 行数 | 通常 1 行（每月一条） |
| 时间连续性 | 上个月是 `2026-03`，本月必须是 `2026-04`，不能跳、不能重 |

**时间连续性检查：**

| 情况 | 结果 |
|------|------|
| 上月 `2026-03`，新数据 `2026-04` | ✅ 正常 |
| 上月 `2026-03`，新数据 `2026-05` | ❌ `TimeGapError`（跳了一个月） |
| 上月 `2026-03`，新数据 `2026-03` | ❌ `TimeOrderError`（重复） |
| 不想报错，只想警告 | 初始化设 `strict=False` |

---

## 4. 吐出什么结果（feed 的输出）

```python
{
    "RF": {
        "time":  ["2026-05", "2026-06", "2026-07"],      # 未来 3 个月
        "pred": [      1043,       1144,       1376],      # 对应的预测值
    },
    "ETS": {
        "time":  ["2026-05", "2026-06", "2026-07"],
        "pred": [      1106,       1365,       1016],
    },
    # 某个模型出错时：
    "ARIMA": {
        "error": "Model was fitted with X, so X_future must be provided.",
    },
}
```

| 字段 | 说明 |
|------|------|
| `time` | 预测对应的时间标签，`["YYYY-MM", ...]` |
| `pred` | 预测值，长度 = horizon（训练时设定，通常 3 或 6） |
| `pred[0]` | **下一个月**的预测值 |
| `pred[1]` | 下两个月的预测值 |
| `error` | 模型出错时的错误信息 |

---

## 5. 完整流程（从启动到每月调用）

### 首次启动（只做一次）

```python
from EpiAI import ModelVault, DeploymentRuntime

# 加载模型
vault = ModelVault.load("/path/to/vault/")

# 初始化运行时
runtime = DeploymentRuntime(vault, time_col="time", time_unit="MS")

# 载入历史数据
runtime.data_table = pd.read_csv("/path/to/history.csv")
```

### 每月调用（重复执行）

```python
# 收到新数据 → feed
new_data = pd.DataFrame({
    "time": ["2026-04-01"],
    "cases": [1500],
})

result = runtime.feed(new_data)

# 解析结果
for name, r in result.items():
    if "error" not in r:
        months = r["time"]          # 时间标签列表
        values = r["pred"]          # 预测值列表
        next_month = values[0]      # 下个月预测值
```

---

## 6. 数据流向图

```
训练阶段（模型开发者）：
  数据 → ForecastPipeline → EpiAITrainer → ModelVault.save("/path/to/vault/")

部署阶段（你）：
  vault = ModelVault.load("/path/to/vault/")          ← 模型
  runtime.data_table = history.csv                    ← 历史数据
  result = runtime.feed(new_data)                     ← 每月新数据
  result["RF"]["pred"][0]                             ← 下个月预测值
```
