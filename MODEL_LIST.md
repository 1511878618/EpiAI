# EpiAI 模型清单

EpiAI 目前支持 **53 个模型**，按算法类型分为以下 4 大类：

---

## 1. 时间序列模型（Time Series — `paradigm='ts'`）

一次性在整个时间序列上训练，预测未来未知值。适用于有强季节性/趋势的传染病数据。

### 1.1 StatsForecast 通用模型（Adapter 包装）

通过 `register_backend()` 统一注册，底层由 `statsforecast` 库驱动。

| 注册名 | 算法 | 说明 |
|--------|------|------|
| `sf_auto_arima` | AutoARIMA | 自动搜索最优 ARIMA(p,d,q)(P,D,Q)ₘ 参数 |
| `sf_arima` | ARIMA | 手动指定参数的 ARIMA 模型 |
| `sf_auto_ets` | AutoETS | 自动搜索最优 ETS 组合（Error/Trend/Seasonal） |
| `sf_auto_ces` | AutoCES | 自动复数指数平滑（Complex Exponential Smoothing） |
| `sf_auto_theta` | AutoTheta | 自动 Theta 方法（M4 竞赛冠军变体） |
| `sf_theta` | Theta | 经典 Theta 方法 |
| `sf_theta_optimized` | OptimizedTheta | 优化版 Theta |
| `sf_theta_dynamic` | DynamicTheta | 动态 Theta |
| `sf_theta_dynamic_optimized` | DynamicOptimizedTheta | 动态优化 Theta |
| `sf_tbats` | TBATS | 三角函数 Box-Cox ARMA 趋势季节模型 |
| `sf_auto_tbats` | AutoTBATS | 自动 TBATS |
| `sf_mstl` | MSTL | 多季节 STL 分解 + 预测 |
| `sf_holt` | Holt | Holt 线性趋势指数平滑 |
| `sf_holt_winters` | Holt-Winters | Holt-Winters 指数平滑（趋势 + 季节） |
| `sf_naive` | Naive | 朴素预测（最后一个观测值重复） |
| `sf_seasonal_naive` | SeasonalNaive | 季节朴素预测（上一年同期值重复） |
| `sf_random_walk_drift` | RandomWalkWithDrift | 随机游走 + 漂移项 |
| `sf_simple_exp_smoothing` | SimpleExpSmoothing | 简单指数平滑 |
| `sf_simple_exp_smoothing_optimized` | OptimizedSimpleExpSmoothing | 优化简单指数平滑 |
| `sf_seasonal_exp_smoothing` | SeasonalExponentialSmoothing | 季节指数平滑 |
| `sf_seasonal_exp_smoothing_optimized` | OptimizedSeasonalExponentialSmoothing | 优化季节指数平滑 |
| `sf_historic_average` | HistoricAverage | 历史均值（常数预测） |
| `sf_window_average` | WindowAverage | 滑动窗口均值 |
| `sf_seasonal_window_average` | SeasonalWindowAverage | 季节滑动窗口均值 |
| `sf_adida` | ADIDA | 间歇需求 ADIDA 法 |
| `sf_croston_classic` | CrostonClassic | 经典 Croston 法（间歇需求） |
| `sf_croston_optimized` | CrostonOptimized | 优化 Croston 法 |
| `sf_croston_sba` | CrostonSBA | Croston SBA 法 |
| `sf_imapa` | IMAPA | 间歇需求 IMAPA 法 |
| `sf_tsb` | TSB | Teunter-Syntetos-Babai 法 |
| `sf_garch` | GARCH | 广义自回归条件异方差模型 |
| `sf_arch` | ARCH | 自回归条件异方差模型 |
| `sf_auto_regressive` | AutoRegressive | 自回归模型（AR） |

### 1.2 内置时间序列模型

EpiAI 原生实现的 TS 模型（无外部依赖或依赖 `statsmodels` / `fbprophet`）。

| 注册名 | 算法 | 说明 |
|--------|------|------|
| `arima` | ARIMA | 基于 statsmodels 的 SARIMAX |
| `ets` | ETS | 误差-趋势-季节指数平滑 |
| `prophet` | Prophet | Facebook Prophet（季节分解 + 变点检测） |
| `serfling` | Serfling | 傅里叶季节回归模型（传染病基线专用） |
| `stlm` | STL+Model | STL 分解 + 任意基模型预测 |
| `bsts` | BSTS | 贝叶斯结构时间序列模型 |

---

## 2. 机器学习模型（sklearn — `paradigm='sklearn'`）

滑动窗口模式：用过去 `lookback` 步的特征预测未来 `horizon` 步。适用于捕捉非线性模式。

| 注册名 | 算法 | 说明 |
|--------|------|------|
| `rf` | Random Forest | 随机森林回归（带 SHAP 可解释性） |
| `xgb` | XGBoost | XGBoost 回归（带 SHAP 可解释性） |
| `lgbm` | LightGBM | LightGBM 回归（MultiOutputRegressor 包装） |
| `svr` | SVR | 支持向量回归（RBF 核） |
| `linearreg` | Linear Regression | 线性回归（基线模型） |

---

## 3. 深度学习模型（torch — `paradigm='torch'`）

滑动窗口模式，PyTorch 驱动，自动 GPU 加速。

| 注册名 | 算法 | 说明 |
|--------|------|------|
| `mlp` | MLP | 多层感知机 |
| `lstm` | LSTM | 长短期记忆网络 |
| `cnn` | CNN | 1D 卷积神经网络 |
| `cnn-lstm` | CNN + LSTM | CNN 特征提取 + LSTM 时序建模 |
| `tcn` | TCN | 时序卷积网络（Temporal Convolutional Network） |
| `transformer` | Transformer | Transformer Encoder 架构 |
| `resnet` | ResNet | 残差网络（1D） |
| `timesnet` | TimesNet | TimesNet（多周期时序分解） |
| `dlinear` | DLinear | 序列分解线性模型（DLinear 风格） |
| `autoformer` | Autoformer | Autoformer（自相关机制） |

---

## 4. 模型命名规则

```
┌ 前缀（来源标识）
│
├ sf_        StatsForecast 后端
├ 无前缀      EpiAI 原生实现
│
└ 算法名      snake_case，全小写
```

- **TS 模型**：构造函数统一接受 `season_length` 参数
- **窗口模型（sklearn/torch）**：构造函数统一接受 `lookback`, `horizon`, `input_dim`, `target_dim`
- **所有模型**：注册名通过 `from EpiAI.models import list_models` 可查
