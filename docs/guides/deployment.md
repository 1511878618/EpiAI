# 部署模块指南

## 功能介绍

部署模块位于 `EpiAI.inference`，包含三个层次：单模型推理、多模型管理、生产部署。

### 三层结构

```
InferencePipeline  →  单模型推理 / 持久化
ModelVault         →  多模型存储 / 对比 / 批量推理
DeploymentRuntime  →  生产部署 / data_table / predict / feed
EpiAI.risk         →  风险预警（独立模块）
```

### 快速路径

```python
from EpiAI.inference import DeploymentRuntime, ModelVault

# 加载 vault + 历史数据
runtime = DeploymentRuntime(
    vault=ModelVault.load("/path/to/vault/"),
    data_table=pd.read_csv("/path/to/history.csv"),
)

# 预测未来 12 个月
preds = runtime.predict(horizon=12)  # → pd.DataFrame
```

---

## InferencePipeline

一个训练好的模型 + 对应的 transforms，提供 `predict(df)` 和 `forecast(steps)`。

| 方法 | 说明 |
|------|------|
| `predict(df)` | 用新数据预测（窗口模型），自动 transform + inverse |
| `forecast(steps)` | 纯未来预测（TS 模型），自动 inverse |
| `update(y_new)` | 在线更新 TS 模型状态 |
| `save(path)` / `load(path)` | 持久化 |

---

## ModelVault

管理多个 InferencePipeline 的集合。

| 方法 | 说明 |
|------|------|
| `from_results(results, bundle)` | 从训练结果创建 |
| `summary()` | 模型对比表（按 Pearson r 排序） |
| `best(metric="PearsonR")` | 最优模型名 |
| `predict_all(new_data, steps)` | 批量推理 |
| `save(path)` / `load(path)` | 持久化 |

---

## DeploymentRuntime

生产部署核心，管理 data_table + predict。

| 方法 | 说明 |
|------|------|
| `predict(horizon=1)` | 预测未来 horizon 步 → pd.DataFrame |
| `feed(new_data)` | 追加新观测 |
| `update_ts(names=None)` | TS 模型滑动窗口重训 |
| `retrain_all()` | 全部模型用全量数据重训 |
| `save(path)` / `load(path)` | 持久化 |

**关键设计：**

- `data_table` 存储**原始尺度**数据
- 模型内部的 `InferencePipeline` 自动处理 transform/inverse
- `predict()` 返回 `pd.DataFrame`（行=时间，列=模型）

---

## EpiAI.risk 风险预警

独立模块，不依赖 DeploymentRuntime。

| 类 | 方法 | 说明 |
|----|------|------|
| `RiskScorer(method)` | `.fit(history)`, `.score_df(preds)` | 预测值→风险等级 |
| `WarningRule(ensemble)` | `.evaluate(risk_df)`, `.report()` | 多模型融合+预警 |

支持四种评分方法：`quantile` / `threshold` / `zscore` / `pct_change`。

---

## 迁移指南（v0.4 → v0.5）

| 旧 API | 新 API |
|--------|--------|
| `runtime.feed(data)["model"]["pred"]` | `runtime.feed(data)` + `runtime.predict()` |
| `runtime.predict(target_time)` | `runtime.predict(horizon=N)` |
| `runtime.predict_range(s, e)` | `runtime.predict(horizon=N)`（已移除循环） |
| `_train_end_time` | `_history_end_time`（自动推断） |
| `results[name]` (dict) | `results[name]` (pd.Series) |
