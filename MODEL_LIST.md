# EpiAI 支持模型一览

## 1. 时间序列模型（一次性训练，预测未来未知值）

### 1.1 StatsForecast 后端

| 算法 | 类别 | 外生变量 | 多目标 | 说明 | 注册名 |
|------|------|---------|--------|------|--------|
| AutoARIMA | 自动模型搜索 | ✓ | - | 自动搜索最优 ARIMA(p,d,q)(P,D,Q)ₘ 参数 | `sf_auto_arima` |
| ARIMA | 自回归移动平均 | ✓ | - | 手动指定参数的 ARIMA | `sf_arima` |
| AutoRegressive | 自回归 | ✓ | - | 纯自回归模型（AR） | `sf_auto_regressive` |
| AutoETS | 自动模型搜索 | - | - | 自动搜索最优 ETS（Error/Trend/Seasonal）组合 | `sf_auto_ets` |
| AutoCES | 自动模型搜索 | - | - | 复数指数平滑（Complex ES） | `sf_auto_ces` |
| Holt | 指数平滑 | - | - | Holt 线性趋势指数平滑 | `sf_holt` |
| Holt–Winters | 指数平滑 | - | - | 趋势 + 季节的经典 Holt–Winters | `sf_holt_winters` |
| SimpleExpSmoothing | 指数平滑 | - | - | 简单指数平滑 | `sf_simple_exp_smoothing` |
| SimpleExpSmoothing (opt.) | 指数平滑 | - | - | 优化版简单指数平滑 | `sf_simple_exp_smoothing_optimized` |
| SeasonalExpSmoothing | 指数平滑 | - | - | 季节指数平滑 | `sf_seasonal_exp_smoothing` |
| SeasonalExpSmoothing (opt.) | 指数平滑 | - | - | 优化版季节指数平滑 | `sf_seasonal_exp_smoothing_optimized` |
| Theta | 统计方法 | - | - | 经典 Theta 法（M4 竞赛变体） | `sf_theta` |
| Theta (optimized) | 统计方法 | - | - | 优化版 Theta | `sf_theta_optimized` |
| DynamicTheta | 统计方法 | - | - | 动态 Theta | `sf_theta_dynamic` |
| DynamicTheta (opt.) | 统计方法 | - | - | 动态优化 Theta | `sf_theta_dynamic_optimized` |
| AutoTheta | 自动模型搜索 | - | - | 自动 Theta 法 | `sf_auto_theta` |
| TBATS | 多季节模型 | - | - | 三角函数 Box-Cox ARMA 趋势季节 | `sf_tbats` |
| AutoTBATS | 多季节模型 | - | - | 自动 TBATS | `sf_auto_tbats` |
| MSTL | 多季节模型 | - | - | 多季节 STL 分解 + 预测 | `sf_mstl` |
| GARCH | 波动率 | - | - | 广义自回归条件异方差 | `sf_garch` |
| ARCH | 波动率 | - | - | 自回归条件异方差 | `sf_arch` |
| Naive | 基线 | - | - | 最后观测值重复 | `sf_naive` |
| SeasonalNaive | 基线 | - | - | 上年同期值重复 | `sf_seasonal_naive` |
| RandomWalkWithDrift | 基线 | - | - | 随机游走 + 漂移 | `sf_random_walk_drift` |
| HistoricAverage | 基线 | - | - | 历史均值（常数预测） | `sf_historic_average` |
| WindowAverage | 基线 | - | - | 滑动窗口均值 | `sf_window_average` |
| SeasonalWindowAverage | 基线 | - | - | 季节滑动均值 | `sf_seasonal_window_average` |
| CrostonClassic | 间歇需求 | - | - | 经典 Croston 法 | `sf_croston_classic` |
| CrostonOptimized | 间歇需求 | - | - | 优化 Croston 法 | `sf_croston_optimized` |
| CrostonSBA | 间歇需求 | - | - | Croston SBA 法 | `sf_croston_sba` |
| ADIDA | 间歇需求 | - | - | ADIDA 法 | `sf_adida` |
| IMAPA | 间歇需求 | - | - | IMAPA 法 | `sf_imapa` |
| TSB | 间歇需求 | - | - | Teunter–Syntetos–Babai 法 | `sf_tsb` |

### 1.2 EpiAI 原生实现

| 算法 | 类别 | 外生变量 | 多目标 | 说明 | 注册名 |
|------|------|---------|--------|------|--------|
| ARIMA（statsmodels） | 自回归移动平均 | ✓ | - | statsmodels 实现的 SARIMAX | `arima` |
| ETS | 指数平滑 | - | - | 误差–趋势–季节指数平滑 | `ets` |
| Prophet | 时间序列分解 | ✓ | - | Facebook Prophet（季节 + 变点检测） | `prophet` |
| Serfling | 传染病基线 | - | - | 傅里叶季节回归（专为传染病设计） | `serfling` |
| STL + 任意模型 | 分解+预测 | - | - | STL 分解后调用基模型预测 | `stlm` |
| BSTS | 贝叶斯结构 | - | - | 贝叶斯结构时间序列 | `bsts` |

## 2. 机器学习模型（滑动窗口模式，sklearn）

| 算法 | 外生变量 | 多目标 | 说明 | 注册名 |
|------|---------|--------|------|--------|
| Random Forest | ✓ 特征输入 | - | 随机森林回归，带 SHAP 可解释性 | `rf` |
| XGBoost | ✓ 特征输入 | - | XGBoost 回归，带 SHAP 可解释性 | `xgb` |
| LightGBM | ✓ 特征输入 | - | LightGBM 回归（MultiOutputRegressor） | `lgbm` |
| SVR | ✓ 特征输入 | - | 支持向量回归（RBF 核） | `svr` |
| Linear Regression | ✓ 特征输入 | - | 线性回归（基线） | `linearreg` |

> 窗口模型的设计即接受多特征输入（lookback × 特征数），因此天然支持外生变量。

## 3. 深度学习模型（滑动窗口模式，PyTorch）

| 算法 | 外生变量 | 多目标 | 说明 | 注册名 |
|------|---------|--------|------|--------|
| MLP | ✓ 特征输入 | - | 多层感知机 | `mlp` |
| LSTM | ✓ 特征输入 | - | 长短期记忆网络 | `lstm` |
| CNN | ✓ 特征输入 | - | 1D 卷积神经网络 | `cnn` |
| CNN–LSTM | ✓ 特征输入 | - | CNN 特征 + LSTM 时序 | `cnn-lstm` |
| TCN | ✓ 特征输入 | - | 时序卷积网络（TCN） | `tcn` |
| Transformer | ✓ 特征输入 | - | Transformer Encoder 架构 | `transformer` |
| ResNet | ✓ 特征输入 | - | 残差网络（1D） | `resnet` |
| TimesNet | ✓ 特征输入 | - | 多周期时序分解网络 | `timesnet` |
| DLinear | ✓ 特征输入 | - | 序列分解线性模型 | `dlinear` |
| Autoformer | ✓ 特征输入 | - | 自相关机制时序模型 | `autoformer` |

---

**表格说明：**

- **外生变量** ✓ = 该模型支持外部特征输入（如气象、人口等辅助变量）
- **多目标** ✓ = 该模型支持同时预测多个疾病/指标（当前全部为单目标）
- 窗口模型（sklearn/torch）天然接受多维特征输入，此处不再逐一标注

**总数：53 个模型**（TS 模型 39 个 + 机器学习 5 个 + 深度学习 10 个）
