# EpiAI 支持模型一览

## 1. 时间序列模型（一次性训练，预测未来未知值）

### 1.1 StatsForecast 后端

| 算法 | 类别 | 说明 | 注册名 |
|------|------|------|--------|
| AutoARIMA | 自动模型搜索 | 自动搜索最优 ARIMA(p,d,q)(P,D,Q)ₘ 参数 | `sf_auto_arima` |
| ARIMA | 自回归移动平均 | 手动指定参数的 ARIMA 模型 | `sf_arima` |
| AutoRegressive | 自回归 | 纯自回归模型（AR） | `sf_auto_regressive` |
| AutoETS | 自动模型搜索 | 自动搜索最优 ETS（Error/Trend/Seasonal）组合 | `sf_auto_ets` |
| AutoCES | 自动模型搜索 | 复数指数平滑（Complex Exponential Smoothing） | `sf_auto_ces` |
| Holt | 指数平滑 | Holt 线性趋势指数平滑 | `sf_holt` |
| Holt–Winters | 指数平滑 | 趋势 + 季节的经典 Holt–Winters 法 | `sf_holt_winters` |
| SimpleExpSmoothing | 指数平滑 | 简单指数平滑 | `sf_simple_exp_smoothing` |
| SimpleExpSmoothing (optimized) | 指数平滑 | 优化版简单指数平滑 | `sf_simple_exp_smoothing_optimized` |
| SeasonalExpSmoothing | 指数平滑 | 季节指数平滑 | `sf_seasonal_exp_smoothing` |
| SeasonalExpSmoothing (optimized) | 指数平滑 | 优化版季节指数平滑 | `sf_seasonal_exp_smoothing_optimized` |
| Theta | 统计方法 | 经典 Theta 方法（M4 竞赛冠军变体） | `sf_theta` |
| Theta (optimized) | 统计方法 | 优化版 Theta | `sf_theta_optimized` |
| DynamicTheta | 统计方法 | 动态 Theta | `sf_theta_dynamic` |
| DynamicTheta (optimized) | 统计方法 | 动态优化 Theta | `sf_theta_dynamic_optimized` |
| AutoTheta | 自动模型搜索 | 自动 Theta 方法 | `sf_auto_theta` |
| TBATS | 多季节模型 | 三角函数 Box-Cox ARMA 趋势季节模型 | `sf_tbats` |
| AutoTBATS | 多季节模型 | 自动 TBATS | `sf_auto_tbats` |
| MSTL | 多季节模型 | 多季节 STL 分解 + 预测 | `sf_mstl` |
| GARCH | 波动率模型 | 广义自回归条件异方差 | `sf_garch` |
| ARCH | 波动率模型 | 自回归条件异方差 | `sf_arch` |
| Naive | 基线 | 最后观测值重复（基线） | `sf_naive` |
| SeasonalNaive | 基线 | 上年同期值重复（季节基线） | `sf_seasonal_naive` |
| RandomWalkWithDrift | 基线 | 随机游走 + 漂移项 | `sf_random_walk_drift` |
| HistoricAverage | 基线 | 历史均值（常数预测） | `sf_historic_average` |
| WindowAverage | 基线 | 滑动窗口均值 | `sf_window_average` |
| SeasonalWindowAverage | 基线 | 季节滑动窗口均值 | `sf_seasonal_window_average` |
| CrostonClassic | 间歇需求 | 经典 Croston 法 | `sf_croston_classic` |
| CrostonOptimized | 间歇需求 | 优化 Croston 法 | `sf_croston_optimized` |
| CrostonSBA | 间歇需求 | Croston SBA 法 | `sf_croston_sba` |
| ADIDA | 间歇需求 | ADIDA 法 | `sf_adida` |
| IMAPA | 间歇需求 | IMAPA 法 | `sf_imapa` |
| TSB | 间歇需求 | Teunter–Syntetos–Babai 法 | `sf_tsb` |

### 1.2 EpiAI 原生实现

| 算法 | 类别 | 说明 | 注册名 |
|------|------|------|--------|
| ARIMA（statsmodels） | 自回归移动平均 | statsmodels 实现的 SARIMAX | `arima` |
| ETS | 指数平滑 | 误差–趋势–季节指数平滑 | `ets` |
| Prophet | 时间序列分解 | Facebook Prophet（季节分解 + 变点检测） | `prophet` |
| Serfling | 传染病基线 | 傅里叶季节回归模型（专为传染病设计） | `serfling` |
| STL + 任意模型 | 分解+预测 | STL 分解后调用任意基模型预测 | `stlm` |
| BSTS | 贝叶斯结构 | 贝叶斯结构时间序列模型 | `bsts` |

## 2. 机器学习模型（滑动窗口模式）

| 算法 | 说明 | 注册名 |
|------|------|--------|
| Random Forest | 随机森林回归，带 SHAP 可解释性 | `rf` |
| XGBoost | XGBoost 回归，带 SHAP 可解释性 | `xgb` |
| LightGBM | LightGBM 回归，MultiOutputRegressor 包装 | `lgbm` |
| SVR | 支持向量回归（RBF 核） | `svr` |
| Linear Regression | 线性回归（基线模型） | `linearreg` |

## 3. 深度学习模型（滑动窗口模式，PyTorch 驱动）

| 算法 | 说明 | 注册名 |
|------|------|--------|
| MLP | 多层感知机 | `mlp` |
| LSTM | 长短期记忆网络 | `lstm` |
| CNN | 1D 卷积神经网络 | `cnn` |
| CNN–LSTM | CNN 特征提取 + LSTM 时序建模 | `cnn-lstm` |
| TCN | 时序卷积网络（Temporal Convolutional Network） | `tcn` |
| Transformer | Transformer Encoder 架构 | `transformer` |
| ResNet | 残差网络（1D） | `resnet` |
| TimesNet | 多周期时序分解网络 | `timesnet` |
| DLinear | 序列分解线性模型（DLinear 风格） | `dlinear` |
| Autoformer | 自相关机制时序模型 | `autoformer` |

---

**总数：53 个模型**（TS 模型 39 个 + 机器学习 5 个 + 深度学习 10 个）
