# 部署管道设计 v2

> 根据讨论后的修订版。核心变化：`DeploymentRuntime` 维护统一数据表，
> 每个模型从中按需取数据，TS 模型默认 align 模式，所有状态持久化到磁盘。

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    DeploymentRuntime                        │
│                                                             │
│  data_table (持久化 DataFrame, 存磁盘)                        │
│  ┌──────┬──────┬──────┬──────┐                              │
│  │ time │cases │temp  │humid │  ← 所有历史数据（特征+目标）  │
│  ├──────┼──────┼──────┼──────┤                              │
│  │ ...  │ ...  │ ...  │ ...  │                              │
│  │ 2026 │ 1200 │ 22.5 │ 0.6  │  ← feed(new_row) 追加        │
│  └──────┴──────┴──────┴──────┘                              │
│                       │                                     │
│          ┌────────────┼────────────┐                        │
│          ▼            ▼            ▼                        │
│       window_1     window_2     TS_model                    │
│       lookback=12  lookback=6   horizon=3                   │
│       horizon=3    horizon=2    ts_mode="align"             │
│          │            │            │                        │
│          └────────────┴────────────┘                        │
│                       ▼                                     │
│       结果: {模型名: (时间轴, 预测值)}                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心概念

### 2.1 统一数据表

`DeploymentRuntime` 内部维护一张持久化的 DataFrame，包含所有历史数据：

```
data_table:
   time     | 登革热  |  t2m_mean  |  tcc_mean  |  province  | ...
  ──────────┼───────┼──────────┼──────────┼───────────┼────
   2024-01  |  500  |   22.5   |   0.6    |   广东    |
   2024-02  |  300  |   23.1   |   0.5    |   广东    |
   ...      |  ...  |   ...    |   ...    |   ...     |
   2026-03  | 1200  |   28.5   |   0.7    |   广东    |
              ↑ feed(new_data) 追加到此
```

- 新增、修改都在这个表上操作
- 每个模型从 table 中取自己需要的列和行
- 表本身持久化到磁盘（parquet / feather 格式）

### 2.2 每个模型按需取数

| 模型类型 | 从 data_table 取什么 |
|---------|---------------------|
| 窗口模型（torch/sklearn） | `table.tail(lookback)[feature_cols]` → 构造 (1, lookback, n_features) |
| 时序模型（TS） | `table.tail(新数据行数)[target_cols]` → `update(y_new)` |

### 2.3 模型各自的 horizon

每个模型训练时保存了自己的 `horizon`：

```python
# RF 模型: horizon=3
# LSTM 模型: horizon=6
# ETS 模型: horizon=3

# 推理时各自输出不同长度：
{
  "RF":   {"time": [2026-05, 2026-06, 2026-07],           "pred": [1200, 800, 600]},
  "LSTM": {"time": [2026-05, 2026-06, 2026-07, 2026-08, 2026-09, 2026-10],
                                                           "pred": [1100, 900, 700, 500, 300, 200]},
  "ETS":  {"time": [2026-05, 2026-06, 2026-07],           "pred": [1150, 850, 650]},
}
```

---

## 3. feed() 内部流程

```python
def feed(self, new_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Parameters
    ----------
    new_data : pd.DataFrame
        新到达的数据行（1 行或多行），必须包含 time_col 和所有特征/目标列

    Returns
    -------
    dict
        {模型名: {"time": 时间轴列表, "pred": 预测值数组}}
    """
```

### 3.1 时间连续性检查

```python
def _check_time_continuity(self, new_data):
    new_times = pd.to_datetime(new_data[self.time_col])
    
    if self.data_table.empty:
        # 首次 feed：检查是否紧接训练结束时间
        expected = self._train_end_time + self._time_delta
        if new_times[0] != expected:
            raise TimeGapError(
                f"首次 feed 时间 {new_times[0]} 不连续于 "
                f"训练结束时间 {self._train_end_time}。"
                f"期望 {expected}，间隔 {new_times[0] - expected}"
            )
    else:
        last = pd.to_datetime(self.data_table[self.time_col].iloc[-1])
        expected = last + self._time_delta

        # 检查每个新行是否连续
        for i, t in enumerate(new_times):
            expected_i = expected + i * self._time_delta
            if t != expected_i:
                raise TimeGapError(
                    f"时间不连续：上次 {last}，期望 {expected_i}，"
                    f"实际 {t}。缺失 {expected_i} ~ {t - self._time_delta}"
                )
    
    # 检查是否乱序（新数据时间不能 ≤ 表中最新时间）
    if not self.data_table.empty:
        last = pd.to_datetime(self.data_table[self.time_col].iloc[-1])
        if new_times[0] <= last:
            raise TimeOrderError(
                f"时间乱序：表中最新 {last}，新数据 {new_times[0]}"
            )
```

**严格模式（默认）：** 任何缺口或乱序直接抛异常。
**宽松模式（`strict=False`）：** 警告但不阻塞。

### 3.2 追加数据

```python
self.data_table = pd.concat(
    [self.data_table, new_data], ignore_index=True
)
```

### 3.3 模型推理

TS 模型默认不自动 update，只 forecast。窗口模型正常 predict。
显式 update 作为独立接口。

```python
results = {}
for name, inferer in self.vault.models.items():
    try:
        # ── 窗口模型（torch / sklearn）───────────
        if inferer.paradigm != "ts":
            lookback = inferer.lookback
            if len(self.data_table) < lookback:
                raise BufferError(
                    f"{name} 需要 {lookback} 行历史，"
                    f"data_table 只有 {len(self.data_table)} 行"
                )
            X_batch = (
                self.data_table[inferer.feature_names]
                .tail(lookback)
                .values[np.newaxis, :, :]
            )
            raw = inferer.predict(X_batch)

            last_time = pd.to_datetime(self.data_table[self.time_col].iloc[-1])
            future_times = pd.date_range(
                start=last_time + self._time_delta,
                periods=inferer.horizon,
                freq=self._time_delta,
            )
            results[name] = {"time": future_times, "pred": raw[0, :, 0]}

        else:
            # ── TS 模型 ───────────────────────────
            # 默认不 update，基于当前已积累的状态 forecast
            raw = inferer.forecast(inferer.horizon)

            last_time = pd.to_datetime(self.data_table[self.time_col].iloc[-1])
            future_times = pd.date_range(
                start=last_time + self._time_delta,
                periods=inferer.horizon,
                freq=self._time_delta,
            )
            results[name] = {"time": future_times, "pred": raw[:, 0, 0]}

    except Exception as e:
        results[name] = {"error": str(e)}

return results
```

### 3.4 显式模型更新（独立接口）

```python
def update_model(self, name: str, new_y: np.ndarray) -> None:
    """显式更新单个 TS 模型的内部状态。

    由用户在确认数据质量后手动调用。后续可扩展为定期批量更新机制。
    """
    inferer = self.vault.get(name)
    if inferer.paradigm != "ts":
        raise TypeError(f"{name} 不是 TS 模型，不支持 update。")

    # 备份当前状态（回滚用）
    backup_dir = self._path / "ts_backup" / name
    backup_dir.mkdir(parents=True, exist_ok=True)
    np.save(backup_dir / f"y_history_{self._feed_count}.npy",
            inferer.model.y_history_)

    # 执行更新
    inferer.update(np.asarray(new_y, dtype=np.float32))
    self._persist()

def update_all_ts(self, data: pd.DataFrame) -> None:
    """批量更新所有 TS 模型。为未来自动重训/增量学习预留接口。"""
    for name, inferer in self.vault.models.items():
        if inferer.paradigm == "ts":
            y_new = data[inferer.target_names].values.ravel().astype(np.float32)
            self.update_model(name, y_new)
```

每次 feed 后自动保存：

```python
def _persist(self):
    # 1. 保存 data_table
    self.data_table.to_parquet(self._path / "data_table.parquet")
    
    # 2. 备份旧的 TS 模型状态（回滚用）
    for name in self.vault.models:
        if self.vault[name].paradigm == "ts":
            # 备份当前 y_history_ 到磁盘
            backup_dir = self._path / "ts_backup" / name
            backup_dir.mkdir(parents=True, exist_ok=True)
            np.save(backup_dir / f"y_history_{self._feed_count}.npy",
                    self.vault[name].model.y_history_)
    
    # 3. 保存 vault（含所有模型）
    self.vault.save(str(self._path / "vault"))
    
    # 4. 保存 runtime 元数据
    meta = {
        "feed_count": self._feed_count,
        "last_time": str(self.data_table[self.time_col].iloc[-1]),
        "n_rows": len(self.data_table),
    }
    (self._path / "runtime_meta.json").write_text(json.dumps(meta))
```

---

## 4. 状态持久化

### 4.1 磁盘结构

```
/tmp/dengue_deployment/
├── runtime_meta.json              # feed 计数、最新时间
├── data_table.parquet             # 全部历史数据
├── vault/                         # ModelVault 目录（manifest + 各模型）
│   ├── manifest.json
│   ├── RF/
│   │   ├── model.zip
│   │   └── meta.json
│   ├── ETS/
│   │   ├── model.zip
│   │   └── meta.json
│   └── ...
└── ts_backup/                     # TS 模型历史状态备份
    ├── ETS/
    │   ├── y_history_0.npy        # 第 0 次 feed 前的状态
    │   ├── y_history_1.npy        # 第 1 次 feed 前的状态
    │   └── ...
    └── ARIMA/
        ├── y_history_0.npy
        └── ...
```

### 4.2 启动恢复

```python
@classmethod
def load(cls, path):
    """从磁盘恢复完整运行时状态。"""
    path = Path(path)
    meta = json.loads((path / "runtime_meta.json").read_text())
    data_table = pd.read_parquet(path / "data_table.parquet")
    vault = ModelVault.load(str(path / "vault"))
    return cls(vault=vault, data_table=data_table, **meta)
```

---

## 5. 各组件关系

```
EpiAITrainer.fit(bundle)
    │
    ▼
InferencePipeline    ← 单模型的一次性推理（离线）
    │
    ▼
ModelVault           ← 多模型管理（存/取/对比）
    │
    ▼
DeploymentRuntime    ← 生产部署（统一数据表 + 时间检查 + 持久化）
```

三者的关系是**递进但不互斥**：
- `InferencePipeline` 适合离线实验一次预测
- `ModelVault` 适合模型比较和归档
- `DeploymentRuntime` 适合生产环境每日 feed

---

## 6. 时间连续性规则总结

| 场景 | 行为 |
|------|------|
| 新数据紧接表中最后一条 | ✅ 正常追加 |
| 新数据与最后一条之间有缺口 | ❌ `TimeGapError` |
| 新数据时间 ≤ 表中最后时间 | ❌ `TimeOrderError` |
| 首次 feed 不接训练结束时间 | ❌ `TimeGapError` |
| 新数据到达但 data_table 行数 < lookback | ❌ `BufferError` |
| strict=False 模式 | ⚠️ 缺口/乱序仅警告，不阻塞 |

---

## 7. 待实现清单

| 功能 | 优先级 |
|------|--------|
| `DeploymentRuntime` 核心类 + `feed()` | P0 |
| 时间连续性检查 | P0 |
| data_table 持久化（parquet） | P0 |
| TS 模型状态备份 | P1 |
| `align` 模式（时间轴自动对齐） | P1 |
| `strict=False` 宽松模式 | P2 |
| Dashboard 集成（输出 → 绘图） | P2 |
