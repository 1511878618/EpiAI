# 框架架构

---

## 整体分层

```
┌──────────────────────────────────────────────────────────────┐
│                         用户代码                              │
│  get("LSTM")(...) / EpiAITrainer.fit(bundle)                 │
│  vault.best("R2") / runtime.feed(new_data)                   │
├──────────────────────────────────────────────────────────────┤
│                     InferenceLayer                            │
│  DeploymentRuntime  │  ModelVault  │  InferencePipeline       │
├──────────────────────────────────────────────────────────────┤
│                     EpiAITrainer                              │
│  torch (epoch循环)  │  sklearn (一次 fit)  │  ts (rolling)    │
├──────────────────────────────────────────────────────────────┤
│                     PipelineBundle                            │
│  train/val/test_x/y + train/val/test_df + get_y_series()     │
├──────────────────────────────────────────────────────────────┤
│                     ForecastPipeline                          │
│  Load → Split → Transform → Window                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 关键设计

### 数据管道

`ForecastPipeline` 将 CS V/Feather 原始数据经过四个阶段转换为模型可用的 3D 窗口数组：

```
原始数据 → TimeSeriesData（时间列 + 特征列 + 目标列 + 实体列）
         → SplitStrategy（TimeSplit / EntitySplit / 等）
         → Transform.Compose（对数 / 标准化 / 时间特征 / 滞后）
         → SlidingWindow（lookback=N, horizon=M）
         → PipelineBundle（3D 数组 + 原始 DataFrame）
```

val/test 拆分在创建窗口时自动拼接前一个 split 尾部的 `lookback + horizon - 1` 行为上下文，确保首步预测覆盖全部测试点。

### 三族模型

所有模型继承 `BaseForecaster`。EpiAITrainer 根据 `paradigm` 自动路由：

| 族 | 训练路径 | 预测路径 |
|----|---------|---------|
| torch | epoch 循环 + AdamW + EarlyStopping | `forward()` + `no_grad()` |
| sklearn | 一次 `fit(X, y)` | 一次 `predict(X)` |
| ts | `fit_sequence(y)` | `predict_sequence(y)` / `forecast(n)` |

### 训练结果

`TrainResult` 统一反标准化和指标计算：

- 窗口模型用 `test_y[:, 0, :]` 作为 y_true（首步）
- TS 模型用 `get_y_series("test")[:n]`
- 指标只比首步（`y_pred[:, 0, :]`），不混拼 horizon 步
- 逆变换处理列对齐

### 推理部署

```
InferencePipeline  →  单模型推理         →  save/load zip
ModelVault         →  多模型管理 + 对比   →  save/load 目录
DeploymentRuntime  →  生产部署 + 逐月 feed → 持久化 data_table
```

---

## 文件结构

```
src/EpiAI/
├── dataset/
│   ├── base.py          ← Transform / SplitStrategy / Compose ABC
│   ├── container.py     ← TimeSeriesData / SplitResult / WindowBundle
│   ├── loaders.py       ← CsvLoader / FeatherLoader / TensorLoader
│   ├── splits.py        ← 6 种拆分策略
│   ├── transforms.py    ← 8 种变换 + SlidingWindow
│   └── pipeline.py      ← ForecastPipeline → PipelineBundle
├── models/
│   ├── base.py          ← BaseForecaster + TorchMixin / SklearnMixin / TSMixin
│   ├── registry.py      ← @register / get / list_models
│   ├── torch_models/    ← 10 个模型（@register + TorchMixin）
│   ├── sklearn_models/  ← 6 个模型（@register + SklearnMixin）
│   └── ts_models/       ← 2 个模型（@register + TSMixin）
├── trainer.py           ← EpiAITrainer + inverse_predictions + _compute_metrics
├── inference.py         ← InferencePipeline + ModelVault + DeploymentRuntime
├── losses.py            ← 爆发感知损失（保持不动）
├── train.py             ← 旧训练循环（保持不动）
└── time_serie_task.py   ← 旧调度器（保持不动）
```

---

## 已修复的数据契约 Bug

| 问题 | 修复 |
|------|------|
| 列不对齐 | `self.mean_[cols].values` 过滤 |
| 窗口 y_true 不对齐 | 窗口用 `test_y[:,0,:]`，TS 用 `get_y_series` |
| horizon 混比 | 只用首步 `y_pred[:, 0, i]` |
| 测试集未覆盖尾行 | 上下文 = `lookback + horizon - 1` |
| 推理时缺少目标列 | `apply_features_only()` |
| ARIMA 外生变量泄漏 | 剥离 X 中与 target 重叠的列 |
