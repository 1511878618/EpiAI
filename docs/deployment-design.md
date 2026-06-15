# 模型部署与实时数据管道设计

> 当模型部署到生产环境后，如何持续接收新数据、维持模型状态、
> 保证时间连续性、输出预测结果。

---

## 1. 生产环境数据流

```
实时数据源（疾控中心 / API / 手动录入）
    │
    ▼
┌──────────────────────────────────────────┐
│            DeploymentRuntime             │
│                                          │
│  每批次新数据到达 → feed(new_df)          │
│    ├─ 检查时间连续性                       │
│    ├─ 更新滚动 buffer                      │
│    ├─ TS 模型: update(y) → forecast(k)    │
│    ├─ 窗口模型: buffer → predict(X)        │
│    └─ 输出: {模型名: (时间轴, 预测值)}     │
└──────────────────────────────────────────┘
    │
    ▼
  Dashboard / 报表 / 预警
```

---

## 2. 三族模型的更新模式

### 2.1 窗口模型（torch / sklearn）

```
         ┌──────────────────────────────┐
buffer:  │ r0 │ r1 │ r2 │ … │ r14 │ r15 │  ← 最后 lookback+horizon-1 行
         └──────────────────────────────┘
                      │ 每次 feed 从 buffer 末尾取 lookback 行
                      ▼
                model.predict(X)  →  (1, horizon, target_dim)
                      │
                      ▼ 结果
                未来 k 步（相对于 buffer 末尾）
```

**特性：**
- 无状态，每次预测独立
- buffer 只用于构造窗口，不进入模型
- 时间连续性由 buffer 的插入顺序保证

### 2.2 时序模型（ARIMA / ETS）

```
模型内部状态:
  y_history_ = [y0, y1, …, yt]
  隐含时间:    [t0, t1, …, tt]

feed(new_y = [yt+1, yt+2]):
  ① update([yt+1, yt+2]) → y_history_ 追加
  ② forecast(3)          → [yt+3, yt+4, yt+5]
```

**特性：**
- 有状态，`y_history_` 是模型的一部分
- `update()` 改变内部状态，影响后续所有预测
- **时间连续性至关重要**——缺口直接导致预测偏移

---

## 3. 时间连续性检查

### 3.1 什么情况下出问题

```
正确:
  t0, t1, t2, t3, t4, t5, …                           → 连续

有缺口:
  t0, t1, t2, t3, t6, t7, …                            → t4, t5 缺失

乱序:
  t0, t1, t2, t5, t3, t4, …                            → 时间回退

新模型上线:
  ──── 训练数据 ————│──────── 部署数据 ─────────────
  训练结束于 tt      新数据从 tt+1 开始                  → 衔接正常

新模型上线（跨越过大）:
  训练结束于 tt      新数据从 tt+10 开始                 → 需要填充
```

### 3.2 连续性规则

| 检查项 | 规则 | 处理方式 |
|--------|------|---------|
| **新数据比最近记录晚 1 步** | 正常 | 直接追加 |
| **新数据与最近记录之间有空缺** | 检测到缺口 | 警告 + 可选择插值或拒绝 |
| **新数据时间 ≤ 最近记录时间** | 乱序 | 错误，拒绝 |
| **新数据是训练结束后第一次到达** | 检查是否紧接训练结束 | 正常（注意上下文衔接） |

### 3.3 元数据设计

每个训练产物需要记录时间元信息：

```python
{
  "time_meta": {
    "train_end": "2026-03-01",         # 训练集最后一条的时间
    "train_start": "2010-01-01",       # 训练集第一条的时间
    "time_unit": "MS",                 # Month Start
    "time_col": "Year/Month",          # 原始时间列名
  }
}
```

TS 模型额外记录 `y_history_` 的时间范围：

```python
{
  "ts_state_meta": {
    "history_start": "2010-01-01",
    "history_end": "2026-03-01",       # 即 train_end
    "n_observations": 195,
  }
}
```

---

## 4. DeploymentRuntime 设计

### 4.1 接口

```python
runtime = DeploymentRuntime(
    vault=model_vault,
    lookback=12,
    horizon=3,
    feature_cols=["t2m_mean", "tcc_mean"],
    time_col="time",
    time_unit="MS",
    strict=True,         # True=时间缺口时报错, False=仅警告
)

# 唯一入口：每批次新数据到达时调用
result = runtime.feed(new_df)

# result = {
#   "RF":     {"time": [2026-05, 2026-06, 2026-07], "pred": [1200, 800, 600]},
#   "ETS":    {"time": [2026-05, 2026-06, 2026-07], "pred": [1100, 900, 700]},
#   "ARIMA":  {"time": [2026-05, 2026-06, 2026-07], "pred": [1150, 850, 650]},
#   "time_meta": {"last_seen": "2026-04-01", "status": "continuous"},
# }
```

### 4.2 feed() 内部流程

```python
def feed(self, new_df: pd.DataFrame) -> Dict[str, Any]:
    # 1. 时间检查
    new_times = new_df[self.time_col]
    _check_continuity(new_times, self._last_time)
    self._last_time = new_times[-1]

    # 2. 更新滚动 buffer
    self._buffer = pd.concat([self._buffer, new_df]).tail(self._buffer_size)

    # 3. 遍历所有模型并行推理
    results = {}
    for name, inferer in self.vault.models.items():
        if inferer.paradigm == "ts":
            # TS 模型: update + forecast
            y_new = new_df[self.target_names[0]].values
            inferer.update(np.asarray(y_new, dtype=np.float32))
            raw = inferer.forecast(self.horizon)
            results[name] = raw[:, 0, 0]
        else:
            # 窗口模型: 从 buffer 构造窗口
            X_batch = self._buffer[self.feature_cols].tail(
                self.lookback
            ).values[np.newaxis, :, :]
            raw = inferer.predict(X_batch)
            results[name] = raw[0, :, 0]

    return self._format_output(results)
```

### 4.3 buffer 管理

```python
# buffer 大小 = lookback + horizon - 1  ← 确保能构造至少一个窗口
# 每次 feed 后自动 trim = buffer.tail(lookback + horizon - 1)
```

但实际部署时，buffer 的最小需求不是固定的：

| 场景 | 需要 buffer 长度 |
|------|----------------|
| 一次预测 horizon 步 | `lookback + horizon - 1` |
| 多次快速 feed（每步 feed 1 行） | `lookback` |
| 用户手动指定更多历史 | 可配置 `buffer_size` |

### 4.4 TS 模型的滚动策略

```python
class DeploymentRuntime:
    ts_mode: Literal["roll", "align"]
        # "roll":  每次 forecast(horizon) → 固定输出 horizon 步
        # "align": 从 buffer 的时间轴自动计算还需要预测多少步
```

**"roll" 模式（标准滚动）：**

```python
# 假设 horizon=3, 时间单位="月"
feed(t=4月数据) → update(y=4月) → forecast(3) → [5月, 6月, 7月]
feed(t=5月数据) → update(y=5月) → forecast(3) → [6月, 7月, 8月]
```

**"align" 模式（自动对齐）：**

```python
# buffer 知道当前最新时间是 4月，下一次预测从 5月开始
feed(t=4月数据) → update(y=4月)
  → 检查 buffer 最新时间 → 5月
  → forecast(k 步，直到 horizon 覆盖)
```

"align" 模式更鲁棒——如果某次 feed 了 2 行数据，`forecast` 自动少预测 1 步，避免时间轴跳变。

---

## 5. 与推理管道的集成

当前现有组件的关系：

```
EpiAITrainer.fit(bundle)
    │
    ▼
InferencePipeline     ← 单个模型的推理（变换→滑窗→预测→反标准化）
    │
    ▼
ModelVault            ← 多模型管理（保存/加载/对比）
    │
    ▼
DeploymentRuntime     ← 生产部署（时间检查/buffer/update/每日 feed）
```

- `InferencePipeline`：负责**一次性的**新数据预测，适用于离线/一次性的场景
- `ModelVault`：负责**多模型的**存储和组织
- `DeploymentRuntime`：负责**持续的**生产 feed，包含时间连续性逻辑

三者是递进关系，不互斥。

---

## 6. 待确认的问题

| 问题 | 选项 |
|------|------|
| 时间缺口时自动插值还是报错？ | 可选 `strict=True/False` |
| buffer 持久化？ | 每次 feed 后保存到磁盘？还是纯内存？ |
| `ts_mode` 的默认值 | `"roll"`（简单直接）还是 `"align"`（更鲁棒）？ |
| 多个模型的不同 horizon？ | 统一用 vault 的配置，或者各自独立？ |
