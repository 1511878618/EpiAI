# 更新日志

## [v0.6.0] — 2026-06-17

### 修复
- `inverse_predictions` 原地修改 `bundle.test_y` 导致重复运行 inf/NaN
- `inverse_predictions` 列名不匹配导致 StandardScaler 逆变换无效
- 预测值写到 `[:, -1, i]`（末步）而非 `[:, 0, i]`（首步）
- 窗口未裁剪，`test_y[:,0,:]` 前 `context_len - lookback` 个值不对应 test_df
- `DateFeatures` 无 `inverse` 阻断逆变换链
- `Identity` 无 `inverse`，且 `transform(self, pd.DataFrame)` 参数名错误
- 教程绘图中 `get_y_series()` 返回标准化值与预测值（原始尺度）不匹配
- 自回归场景 `feature_cols ∩ target_cols` 时特征列为空

### 新增
- `DeploymentRuntime.update_model()` TS 模型显式更新接口
- CI: `.github/workflows/pytest.yml`
- `CONTRIBUTING.md`、`.pre-commit-config.yaml`
- 文档重构：index / quickstart / architecture / data-pipeline + 4 篇模块指南

### 变更
- torch 改为可选依赖 `EpiAI[torch]`
- 测试文件 `test_phaseN` → 功能名
- pyproject.toml v0.5.0

## [v0.5.0] — 2026-06-16

### 新增
- 文档体系重构：index / quickstart / architecture / data-pipeline + 4 篇模块指南
- 每篇指南包含功能介绍 + 扩展指南（如何加新 Transform / Split / 模型 / 范式 / 部署后端）

### 变更
- docs/ 目录重组，旧文档移至 docs/archive/
- `.pre-commit-config.yaml`：black + ruff + trailing-whitespace
- `CONTRIBUTING.md`：开发环境、代码风格、PR 流程
- CI：`.github/workflows/pytest.yml`（Python 3.10–3.12）
- `MANIFEST.in`：打包数据文件
- 测试文件从 `test_phaseN` 重命名为功能名
- `tutorial/tutorial-dengue-1.ipynb` 移除（重复文件）

## [v0.4.0] — 2026-06-15

### 新增
- `DeploymentRuntime`：生产部署运行时，统一 data_table、时间连续性检查、持久化
- `ModelVault`：多模型存储、对比表、批量推理
- `SlidingWindow.apply_features_only()`：推理时只需特征列，无需目标列
- `InferencePipeline`：单模型推理管道（变换→滑窗→预测→反标准化）
- `TorchMixin.predict()`：默认 forward + no_grad + 自动设备对齐
- `EpiAITrainer` 三路路由：torch epoch 循环 / sklearn 一次 fit / TS rolling origin
- 10 个 PyTorch 模型注册：MLP / LSTM / CNN / CNN-LSTM / ResNet / TCN / Transformer / DLinear / Autoformer / TimesNet
- 6 个 sklearn 模型注册：RF / XGB / LGBM / SVR / GLM / TabPFN
- 2 个 TS 模型注册：ARIMA / ETS
- `tutorial-full.ipynb`：完整流程教程（留数据→训练→逐月 feed→对比）

### 修复
- `StandardScaler.transform/inverse` 列对齐：`self.mean_[cols].values` 过滤，避免 Pandas index 对齐产生新列
- `RobustScaler.transform/inverse` 同上
- `_build_result` 中窗口模型的 y_true 改为 `test_y[:, 0, :]`（窗口目标首步）
- `_compute_metrics` 只比 `y_pred[:, 0, i]`（首步），不再混拼所有 horizon 步
- `inverse_predictions` 用 `[:, 0, i]` 而非 `[:, -1, i]`
- val/test 窗口上下文长度改为 `lookback + horizon - 1`，确保首步覆盖全部测试点
- ARIMA 数据泄漏：剥离 X 中与 target 重叠的列，防止未来值作为外生变量
- AdamW kwargs 过滤：只传 lr/weight_decay 等有效参数，block max_epochs/batch_size
- TorchMixin.predict() 输入 tensor 自动 `to(device)`
- TS 模型 forecast 滑动输出：`forecast(horizon + feed_count)[-horizon:]`

## [v0.3.0] — 2026-06-10

### 新增
- `PipelineBundle` DataFrame 方法：`get_y_series()` / `get_X_series()`
- `EpiAITrainer` 统一训练入口
- `@register` / `get()` / `list_models()` 模型注册系统
- `BaseForecaster` + 3 个 Mixin（TorchMixin / SklearnMixin / TSMixin）

### 修复
- FeatureLag 多实体跨边界滞后修复

## [v0.2.0] — 2026-06-05

### 新增
- `ForecastPipeline` 数据管道调度器
- 6 种数据拆分策略（TimeSplit / EntitySplit / EntityTimeSplit / CustomIndexSplit / NoSplit / CrossValidationSplit）
- 8 种数据变换（Identity / StandardScaler / RobustScaler / Log1pTransform / BoxCoxTransform / SelectColumns / DateFeatures / FeatureLag）
- `SlidingWindow` 滑窗生成
- `CsvLoader` / `FeatherLoader` / `TensorLoader` 数据加载器

## [v0.1.0] — 2026-06-01

### 新增
- 项目初始化，基础数据抽象层
- `TimeSeriesData` / `SplitResult` / `WindowBundle` 容器
- `OmicronV2`、`MultiTargetCityDatasetBuilder` 等旧版数据处理类
