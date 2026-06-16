# EpiAI 部署 API 参考

> `inference.py` 中定义的三个核心类和异常，供部署方调用。

---

## 目录

1. [InferencePipeline](#1-inferencepipeline) — 单模型推理
2. [ModelVault](#2-modelvault) — 多模型管理
3. [DeploymentRuntime](#3-deploymentruntime) — 生产部署
4. [异常](#4-异常)

---

## 1. InferencePipeline

封装一个已训练的模型及其变换管道，用于对新数据做预测。

### 构造函数

一般不直接构造，通过以下工厂方法创建：

```python
InferencePipeline.from_train_result(result)
```
- `result: TrainResult` — `EpiAITrainer.fit()` 的返回值

```python
InferencePipeline.from_components(model, transforms, lookback, horizon, feature_names, target_names)
```

### 方法

#### `predict(df) → np.ndarray`

对新数据做预测。适用于 torch / sklearn 窗口模型。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | `pd.DataFrame` | 至少 `lookback` 行，需包含 `feature_names` 列 |

| 返回 | 形状 | 说明 |
|------|------|------|
| `np.ndarray` | `(N, horizon, target_dim)` | 反标准化后的预测值 |

**示例：**
```python
inferer = InferencePipeline.load("model.zip")
new = pd.DataFrame({"cases": [100, 200, ..., 1500]})  # 12 行
pred = inferer.predict(new)     # (1, 3, 1)
```

---

#### `forecast(steps) → np.ndarray`

纯未来预测。仅适用于 TS 模型（ARIMA / ETS）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `steps` | `int` | 预测步数 |

| 返回 | 形状 | 说明 |
|------|------|------|
| `np.ndarray` | `(steps, 1, 1)` | 预测值 |

**示例：**
```python
forecast = inferer.forecast(6)  # 未来 6 个月
```

---

#### `update(y_new) → np.ndarray`

用新观测值更新 TS 模型状态并预测对应时段。仅适用于 TS 模型。

| 参数 | 类型 | 说明 |
|------|------|------|
| `y_new` | `np.ndarray` | 新观测值，`(T,)` 或 `(T, 1)` |

| 返回 | 形状 | 说明 |
|------|------|------|
| `np.ndarray` | `(T, 1, 1)` | 对观测时段的预测 |

---

#### `save(path) → str`

序列化到 zip 文件。包含模型权重、变换管道、配置。

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` 或 `Path` | 输出路径（自动加 `.zip`） |

| 返回 | 说明 |
|------|------|
| `str` | 绝对路径 |

**磁盘内容：**
```
model.zip
├── config.json       # lookback, horizon, feature_names, target_names
├── model.pkl         # pickled BaseForecaster
└── transforms.pkl    # pickled Compose（或空）
```

---

#### `load(path) → InferencePipeline`

从 zip 文件恢复。

---

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `.paradigm` | `str` | `"torch"` / `"sklearn"` / `"ts"` |
| `.lookback` | `int` | 输入窗口长度 |
| `.horizon` | `int` | 预测步数 |
| `.feature_names` | `list[str]` | 特征列名 |
| `.target_names` | `list[str]` | 目标列名 |
| `.model` | `BaseForecaster` | 底层模型实例 |

---

## 2. ModelVault

多模型存储、对比、批量推理。

### 构造函数

```python
ModelVault.from_results(results, bundle)
```
- `results: dict[str, TrainResult]` — `{模型名: TrainResult}`
- `bundle: PipelineBundle` — 训练用的数据管道输出

### 方法

#### `summary() → pd.DataFrame`

返回所有模型的指标对比表，按 R² 降序。

| 返回列 | 说明 |
|--------|------|
| `paradigm` | 模型族 |
| `MAE`, `RMSE`, `MAPE` | 误差指标 |
| `R2` | R² 决定系数 |
| `PearsonR` | 皮尔逊相关系数 |
| `n` | 评估样本数 |

---

#### `best(metric="R2") → str`

返回指定指标最优的模型名。

| 参数 | 默认 | 说明 |
|------|------|------|
| `metric` | `"R2"` | 指标名（`"MAE"`, `"RMSE"`, `"PearsonR"` 等） |

---

#### `predict_all(new_data=None, steps=6) → dict[str, np.ndarray]`

运行所有模型进行推理，返回 `{模型名: 预测值}`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `new_data` | `pd.DataFrame` 或 `None` | 窗口模型需要的特征数据 |
| `steps` | `int` | TS 模型的 forecast 步数 |

内部自动路由：
- torch / sklearn → 调 `predict(new_data)`
- ARIMA / ETS → 调 `forecast(steps)`

---

#### `get(name) → InferencePipeline`

获取指定模型的推理管道。

支持下标访问：`vault["RF"]` 等价于 `vault.get("RF")`。

---

#### `save(path) → str`

保存到目录。每个模型一个子目录。

**磁盘内容：**
```
vault/
├── manifest.json        # 全部模型及指标
├── RF/
│   ├── model.zip        # InferencePipeline 包
│   └── meta.json        # 训练参数
├── ETS/
│   ├── model.zip
│   └── meta.json
└── ...
```

---

#### `load(path) → ModelVault`

从目录恢复。

---

## 3. DeploymentRuntime

生产部署运行时。维护统一历史数据表，每次新数据到达时自动检查时间连续性并运行所有模型。

### 构造函数

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
| `vault` | — | 已训练的模型库 |
| `time_col` | `"time"` | 数据中时间列的名称 |
| `time_unit` | `"MS"` | 时间单位：`"MS"`（月）、`"D"`（日）、`"h"`（时）等 |
| `strict` | `True` | `True` 时时间不连续抛异常；`False` 时仅警告 |

---

### 方法

#### `feed(new_data) → dict[str, dict]`

核心方法。追加新数据、检查连续性、运行所有模型、自动持久化。

| 参数 | 类型 | 说明 |
|------|------|------|
| `new_data` | `pd.DataFrame` | 新到达的数据行，需包含 `time_col` 以及模型需要的特征/目标列 |

**返回结构：**
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

**内部流程：**
1. 检查时间连续性（缺口 / 乱序）
2. 追加到 `data_table`
3. 遍历各模型：
   - 窗口模型：`data_table.tail(lookback)[feature_names]` → `predict()`
   - TS 模型：`forecast(horizon + feed_count)[-horizon:]` — 输出随时间滑动
4. 自动 `save()` 到磁盘

---

#### `update_model(name, y_new) → None`

显式更新单个 TS 模型的内部状态。与 `feed()` 独立，由调用方在确认数据质量后执行。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 模型名（必须在 vault 中） |
| `y_new` | `np.ndarray` | 新观测值 |

更新前自动备份当前 `y_history_` 到 `ts_backup/` 目录。

---

#### `update_all_ts(data) → None`

批量更新所有 TS 模型。预留接口，后续可扩展为自动重训/增量学习。

---

#### `save(path) → str`

持久化全部状态。

```python
runtime.save("/path/to/runtime/")
```

**磁盘内容：**
```
runtime/
├── runtime_meta.json     # feed_count, 最新时间等
├── data_table.parquet    # 全部历史数据（若无 pyarrow 则存 CSV）
├── vault/                # ModelVault 目录
└── ts_backup/            # TS 模型状态快照
```

---

#### `load(path) → DeploymentRuntime`

恢复全部状态（含 data_table 和所有模型）。

---

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `.data_table` | `pd.DataFrame` | 全部历史数据，可读写 |
| `.vault` | `ModelVault` | 模型库 |
| `._feed_count` | `int` | 已 feed 次数 |

---

### 启动配置

```python
# 部署前的准备工作
runtime = DeploymentRuntime(vault, time_col="time", time_unit="MS")

# 载入训练数据作为历史（让窗口模型能有足够 lookback）
runtime.data_table = training_data.copy()

# 设置训练结束时间（首次 feed 会检查是否衔接）
runtime._train_end_time = training_data["time"].iloc[-1]
```

---

## 4. 异常

| 异常 | 继承 | 触发条件 |
|------|------|---------|
| `TimeGapError` | `ValueError` | 新数据与表中最新数据之间存在时间缺口 |
| `TimeOrderError` | `ValueError` | 新数据时间 ≤ 表中最新时间（乱序或重复） |
| `BufferError` | `RuntimeError` | `data_table` 行数不足以满足某模型的 `lookback` |

---

## 5. 快速参考

```python
# 1. 加载模型
vault = ModelVault.load("/path/to/vault/")

# 2. 查看有哪些模型
print(vault.summary())

# 3. 选最优模型
best = vault.best("R2")
inferer = vault[best]

# 4. 对新数据做预测
pred = inferer.predict(new_data)

# 5. 生产部署
runtime = DeploymentRuntime(vault, time_col="time")
runtime.data_table = history
result = runtime.feed(new_observation)
```
