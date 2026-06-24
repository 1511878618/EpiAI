# 模型清单

EpiAI 内置 22 个模型，按范式分为 **时序模型（TS）** 和 **窗口模型（Window）** 两类。

---

## 模型总览

```
范式       | 模型
───────────┼──────────────────────────────────────────────
TS         | ARIMA  BSTS  ETS  Prophet  STLM  Serfling
Window     | RF  XGB  LGBM  SVR  LinearReg  TabPFN
(sklearn)  |
Window     | LSTM  CNN  CNN-LSTM  MLP  ResNet  TCN
(torch)    | Autoformer  DLinear  TimesNet  Transformer
```

---

# TS 模型（时序模型）

**共性特征：**
- 输入：一维时间序列 `(T,)`，不依赖滑窗
- 训练：`fit_sequence(y_train)`，维护内部历史状态
- 推理：`forecast(steps)` 从训练末尾外推
- 滚动评估：`predict_sequence(y_test, update_state=True)` 逐时间步滚动
- 数据量：至少 24 个月（含 1 个完整季节周期），推荐 ≥ 60 个月
- 日期传递：通过 `dates` 参数传递真实日期

---

## ARIMA — 自动 ARIMA

| 项 | 说明 |
|----|------|
| 类 | `AutoARIMAXRollingForecaster` |
| 范式 | TS |
| 引擎 | `pmdarima.auto_arima` 自动选阶 + `pm.ARIMA` 拟合 |
| 核心参数 | `seasonal=True`, `m=12`, `rolling_window_size='all'`, `horizon=1` |

**适用场景：**
- 基线预测，强通用性
- 有明显季节性的月度数据（dengue/flu 等）
- 作为 TS 模型的 benchmark

**优点：** 自动差分/季节选阶，无需手动调参
**缺点：** auto_arima 耗时长（尤其数据量大时）；不支持多目标

---

## BSTS — 贝叶斯结构时间序列

| 项 | 说明 |
|----|------|
| 类 | `BSTSForecaster` |
| 范式 | TS |
| 引擎 | PyMC MCMC（回退：statsmodels MLE） |
| 核心参数 | `seasonal_periods=12`, `niter=250`, `burn=50` |

**适用场景：**
- 需要置信区间/不确定性量化
- 数据量中等（50-200 条）
- 研究级分析，理解 posterior 分布

**优点：** 完整贝叶斯推断；后验预测分布 → 可信区间
**缺点：** MCMC 收敛慢（50-200 条数据需数秒）；需安装 PyMC

---

## ETS — 指数平滑

| 项 | 说明 |
|----|------|
| 类 | `ETSForecaster` |
| 范式 | TS |
| 引擎 | `statsmodels.tsa.holtwinters.ExponentialSmoothing` |
| 核心参数 | `seasonal_periods=12`, `error='add'`, `trend='add'`, `seasonal='add'` |

**适用场景：**
- 快速 baseline，训练极快
- 数据有明确趋势+季节
- 需解释性（error/trend/seasonal 组件可读）

**优点：** 训练毫秒级，参数可解释
**缺点：** 只能加性/乘性季节；长 horizon forecast 收敛到趋势线

---

## Prophet — Facebook Prophet

| 项 | 说明 |
|----|------|
| 类 | `ProphetForecaster` |
| 范式 | TS |
| 引擎 | `prophet.Prophet`（斯坦福/ Meta） |
| 核心参数 | `growth='linear'`, `seasonality_mode='multiplicative'`（病例有季节幅度变化） |

**适用场景：**
- 数据有 changepoint（政策/疫情突变）
- 缺值容忍度高
- 自动处理节假日效应（暂未启用）

**优点：** 强鲁棒性；缺值自动处理；趋势变化点检测
**缺点：** 需安装 prophet（pip install prophet）；对大异常值敏感

---

## STLM — STL 分解 + ARIMA

| 项 | 说明 |
|----|------|
| 类 | `STLMARIMAForecaster` |
| 范式 | TS |
| 引擎 | STL 分解（statsmodels）+ `pm.ARIMA` 拟合剩余 |
| 核心参数 | `seasonal_periods=12`, `seasonal=True` |

**适用场景：**
- 复杂季节模式（非恒定季节幅度）
- 需要将季节与趋势分开建模
- 数据含强季节性

**优点：** STL 分解灵活（容许季节变化），ARIMA 建模剩余
**缺点：** 两步拟合（STL + ARIMA）速度中等

---

## Serfling — Serfling 回归

| 项 | 说明 |
|----|------|
| 类 | `SerflingForecaster` |
| 范式 | TS |
| 引擎 | 线性回归（sklearn） |
| 核心参数 | `seasonal_periods=12`, `fourier_order=3`, `trend_order=1` |

**适用场景：**
- 流行病超额死亡/超额病例估计
- 基线 vs 异常分离（经典 Serfling 方法）
- 解释性要求高（回归系数可解读）

**优点：** 极快，可解释，经典流行病学方法
**缺点：** 线性假设；对非线性趋势拟合差

---

# 窗口模型（Window）

**共性特征：**
- 输入：`(N, lookback, input_dim)` — 批量的滑窗数据
- 每个窗口用过去 L 个时间步预测未来 H 步
- 训练方式：`fit(train_x, train_y)` 监督学习
- 多步预测：一次性输出 horizon 步
- 推理：`predict(x)` 输入 3D 张量

---

# sklearn 窗口模型

## RF — 随机森林

| 项 | 说明 |
|----|------|
| 类 | `RandomForestForecaster` |
| 引擎 | `sklearn.ensemble.RandomForestRegressor`（原生多输出） |
| 默认参数 | `n_estimators=200`, `max_depth=None`, `random_state=42` |

**优点：** 训练并行快，抗过拟合，无需归一化
**缺点：** 模型体积大（200 棵树大数十 MB）；对低频数据泛化弱

---

## XGB — XGBoost

| 项 | 说明 |
|----|------|
| 类 | `XGBSingleForecaster` |
| 引擎 | `xgboost.XGBRegressor`（原生多输出） |
| 默认参数 | `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8` |

**优点：** 树模型 SOTA；GPU 加速；正则化好
**缺点：** 超参敏感；小数据容易过拟合

---

## LGBM — LightGBM

| 项 | 说明 |
|----|------|
| 类 | `LGBMSingleForecaster` |
| 引擎 | `lightgbm.LGBMRegressor` + `MultiOutputRegressor`（多输出需包装） |
| 默认参数 | `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `device='gpu'` |

**注意：** LightGBM sklearn 接口不支持原生多输出回归，内部使用 `MultiOutputRegressor` 创建 H 个独立子模型。训练时间约为 RF/XGB 的 H 倍。

**优点：** GPU 加速；速度快（单模型）；leaf-wise 树
**缺点：** 多输出需 H 倍开销；H 越大越慢

---

## SVR — 支持向量回归

| 项 | 说明 |
|----|------|
| 类 | `SVRForecaster` |
| 引擎 | `sklearn.svm.SVR` + `MultiOutputRegressor` |
| 默认参数 | `kernel='rbf'`, `C=1.0`, `epsilon=0.1` |

**优点：** 小数据泛化好；核方法处理非线性
**缺点：** 需归一化；大数据慢；多输出需 `MultiOutputRegressor`

---

## LinearReg — 线性回归

| 项 | 说明 |
|----|------|
| 类 | `LinearRegForecaster` |
| 引擎 | `sklearn.linear_model.LinearRegression` |
| 默认参数 | `fit_intercept=True` |

**优点：** 极快；可解释（回归系数）
**缺点：** 线性假设；无法捕捉非线性

---

## TabPFN — TabPFN（预训练 Transformer）

| 项 | 说明 |
|----|------|
| 类 | `TabPFNMultiForecaster` |
| 引擎 | TabPFN（预训练小 Transformer） |
| 需要 | 预训练权重文件 `model_path` |

**优点：** 小数据 Few-shot 预测；零调参
**缺点：** 需额外下载权重；推理比传统 ML 慢

---

# torch 窗口模型

**共性特征：**
- 输入归一化：全部模型内置 `nn.LayerNorm(input_dim)`（input_dim>1 时生效）
- 优化器：`AdamW`（trainer 自动配置）
- 损失函数：`MSELoss`（默认）
- 早期停止：patience=10（默认）

---

## LSTM — 长短期记忆网络

| 项 | 说明 |
|----|------|
| 类 | `LSTMForecaster` |
| 结构 | `LSTM(128) → Dropout → Linear(128 → H*T)` |
| 默认参数 | `hidden_dim=128`, `dropout=0.1`, `num_layers=1` |

**适用场景：** 时序预测通用选择；中等数据量（200-2000 样本）
**优点：** 时序建模能力强；已内建 tanh gate → 数值稳定
**缺点：** 训练比 MLP/CNN 慢；对超参较敏感

---

## CNN — 卷积神经网络（时序）

| 项 | 说明 |
|----|------|
| 类 | `CNNForecaster` |
| 结构 | `Conv1d(16) → ReLU → MaxPool → Conv1d(32) → ReLU → MaxPool → Flatten → Linear → Linear` |
| 默认参数 | `dropout=0.5`, `conv1={'hid':16}`, `conv2={'hid':32}` |

**适用场景：** 快速特征提取；捕捉局部模式
**优点：** 训练快；MaxPool + Dropout(0.5) 抗过拟合
**缺点：** 感受野有限（依赖 kernel size + 层数）；对长期依赖弱

---

## CNN-LSTM — 卷积 + LSTM 混合

| 项 | 说明 |
|----|------|
| 类 | `CNNLSTMForecaster` |
| 结构 | `Conv1d → BatchNorm → ReLU → LSTM → Dropout → Linear` |
| 默认参数 | `cnn_channels=32`, `kernel_size=3`, `lstm_hidden=64`, `lstm_layers=1`, `dropout=0.1` |

**适用场景：** 局部特征 + 长期依赖的组合需求
**优点：** CNN 提局部特征 + LSTM 建模时序；BatchNorm 稳定训练
**缺点：** 参数较多，训练相对慢

---

## MLP — 多层感知机

| 项 | 说明 |
|----|------|
| 类 | `MLPForecaster` |
| 结构 | `Linear(L*C → 128) → ReLU → Dropout → Linear(128 → 128) → ReLU → Dropout → Linear(128 → H*T)` |
| 默认参数 | `hidden_dim=128`, `dropout=0.1` |

**适用场景：** 极简 baseline；小数据量快速验证
**优点：** 参数最少，训练最快
**缺点：** 无时间结构建模（flatten 丢时序）

---

## ResNet — 残差网络（时序版）

| 项 | 说明 |
|----|------|
| 类 | `ResNetForecaster` |
| 结构 | `Conv1d → ReLU → BN → ... × num_blocks → Flatten → FC` |
| 默认参数 | `base_channels=64`, `num_blocks=2`, `kernel_size=3` |

**适用场景：** 中等深度模型；需残差恒等映射
**优点：** BatchNorm + 残差 → 训练稳定；可加深
**缺点：** 参数较多；对输入尺度敏感（已加 input_norm）

---

## TCN — 时间卷积网络

| 项 | 说明 |
|----|------|
| 类 | `TCNForecaster` |
| 结构 | `TemporalBlock × channels → 取最后时间步 → Linear(32 → H*T)` |
| 默认参数 | `channels=[32, 32]`, `kernel_size=3` |

**适用场景：** 替代 RNN 的时序建模
**优点：** 因果卷积（不泄漏未来）；空洞卷积大感受野；BatchNorm 稳定
**缺点：** 默认 2 层感受野仅 13 步（需增加 `channels` 长度扩大）

---

## Autoformer — 自相关注意力

| 项 | 说明 |
|----|------|
| 类 | `AutoformerForecaster` |
| 结构 | `Embedding → AutoCorrelation Encoder → Decoder → Projection → last step` |
| 默认参数 | `d_model=256`, `n_heads=8`, `e_layers=2`, `dropout=0.1` |

**适用场景：** 长期预测（horizon ≥ 24）；序列分解需求
**优点：** AutoCorrelation 替代 attention → O(NlogN)；内置序列分解
**缺点：** 参数多（~1M）；对短序列（<100）效果不稳定

---

## DLinear — 简单线性分解

| 项 | 说明 |
|----|------|
| 类 | `DLinearForecaster` |
| 结构 | `SeriesDecomp → 两个 Linear 共享权重 → 相加 → mean` |
| 默认参数 | `kernel_size=3` |

**注：** 不算深度学习，只是一个线性层 + 序列分解。无 activation/norm/dropout。

**适用场景：** 极简 baseline；数据量极少（<50 条）
**优点：** 零超参，训练毫秒
**缺点：** 纯线性，表达力有限

---

## TimesNet — 时序二维化

| 项 | 说明 |
|----|------|
| 类 | `TimesNetForecaster` |
| 结构 | `Embedding → TimesBlock × e_layers → Flatten → Linear` |
| 默认参数 | `d_model=256`, `e_layers=2`, `top_k=3`, `dropout=0.1` |

**适用场景：** 复杂周期模式；多周期叠加（每日+每周+每月）
**优点：** 将 1D 时序转 2D 张量用 Inception 模块处理；Capture 多周期
**缺点：** 参数多；训练比 LSTM 慢

---

## Transformer — 标准 Transformer

| 项 | 说明 |
|----|------|
| 类 | `TransformerForecaster` |
| 结构 | `PositionalEncoding → TransformerEncoder × num_layers → Flatten → Linear` |
| 默认参数 | `d_model=128`, `nhead=8`, `num_layers=3`, `dropout=0.1` |

**适用场景：** 通用 Seq2Seq；数据量大（>1000 条）
**优点：** Self-attention 全局建模；并行训练
**缺点：** 小数据严重过拟合；O(N²) 计算量

---

# 模型选择速查

| 数据量 | 推荐模型 |
|--------|---------|
| < 50 条 | DLinear / LinearReg / Prophet |
| 50-200 条 | ETS / ARIMA / Serfling / RF / XGB / LSTM |
| 200-1000 条 | LSTM / TCN / ResNet / XGB / LGBM / TabPFN |
| > 1000 条 | Autoformer / TimesNet / Transformer / LGBM (GPU) |

| 目标 | 推荐模型 |
|------|---------|
| 快速 baseline | LinearReg / ETS / MLP |
| 最高精度 | XGB / LSTM / TCN / ResNet |
| 不确定性 | BSTS / Prophet |
| 解释性 | LinearReg / Serfling / ETS / RF |
| 超额估计 | Serfling（经典方法） |
| 爆发预警 | 所有 TS 模型（短期） + XGB/LSTM（长期） |

---

# 所有模型共用的输入/输出

| 维度 | 训练输入 | 训练输出 | 推理输入 | 推理输出 |
|------|---------|---------|---------|---------|
| window | `(N, L, F)` | `(N, H, T)` | `(N, L, F)` | `(N, H, T)` |
| TS | `(T,)` 或 `(T, 1)` | `(T,)` | `forecast(steps)` | `(steps, 1, 1)` |

- `N` = 样本数, `L` = lookback, `F` = 特征数
- `H` = horizon, `T` = target_dim（通常为 1: "cases"）
- 所有值已逆变换回**原始尺度**
