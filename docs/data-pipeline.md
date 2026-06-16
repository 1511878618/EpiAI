# 数据管道

数据管道将原始表格数据转换为模型可用的 3D 窗口数组 `(N, lookback, n_features)`。

---

## 流程

```
原始数据 (CSV/Feather)
    │
    ▼ TimeSeriesData
    │  time_col / target_cols / feature_cols / entity_col
    │
    ▼ SplitStrategy
    │  TimeSplit / EntitySplit / EntityTimeSplit /
    │  CustomIndexSplit / NoSplit / CrossValidationSplit
    │
    ▼ Transform.Compose
    │  StandardScaler / Log1pTransform / DateFeatures / FeatureLag / ...
    │  ★ 拟合在训练集，transform 应用到所有 split
    │
    ▼ SlidingWindow
    │  lookback=N, horizon=M
    │
    ▼ PipelineBundle
       train/val/test_x/y (3D) + train/val/test_df + get_y_series()
```

---

## 核心类

### `ForecastPipeline`

编排整个流程。四个阶段分别由 loader、split、transforms、window 参数控制。

```python
from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, Compose,
    StandardScaler, Log1pTransform, DateFeatures,
    FeatureLag, SlidingWindow,
)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="date", target_cols="cases",
                     feature_cols=["temp", "humid"]),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=Compose([
        Log1pTransform(columns=["cases"]),
        StandardScaler(),
        DateFeatures(time_col="date", features=["month", "season"]),
        FeatureLag(columns=["cases"], lags=[1, 3, 6]),
    ]),
    window=SlidingWindow(lookback=12, horizon=3),
).run("data.csv")
```

### `PipelineBundle`

管道的输出，包含训练/验证/测试的窗口数据和原始序列。

| 属性 | 类型 | 说明 |
|------|------|------|
| `train_x/y` | `(N, L, D)` / `(N, H, T)` | 训练窗口 |
| `val_x/y` | `(N, L, D)` / `(N, H, T)` | 验证窗口 |
| `test_x/y` | `(N, L, D)` / `(N, H, T)` | 测试窗口 |
| `train/val/test_df` | `DataFrame` | 变换后的原始序列 |
| `get_y_series(split)` | `ndarray` | TS 模型用的原始 y |
| `get_X_series(split)` | `ndarray` | TS 模型用的原始 X |
| `lookback` / `horizon` | `int` | 窗口参数 |
| `n_features` / `n_targets` | `int` | 维度信息 |

### 数据拆分

| 策略 | 用途 |
|------|------|
| `TimeSplit` | 按时间比例拆分（单实体） |
| `EntitySplit` | 按实体列拆分（按省份/城市） |
| `EntityTimeSplit` | 按实体+时间拆分 |
| `CustomIndexSplit` | 自定义索引拆分 |
| `NoSplit` | 不拆分，全部训练 |
| `CrossValidationSplit` | 时间序列交叉验证 |

### 数据变换

| 变换 | 说明 |
|------|------|
| `StandardScaler` | 标准化 (z-score) |
| `RobustScaler` | 基于分位数的标准化 |
| `Log1pTransform` | log(1+x) 变换 |
| `BoxCoxTransform` | Box-Cox 变换 |
| `DateFeatures` | 提取年月日/季节/节假日 |
| `FeatureLag` | 生成滞后特征 |
| `SelectColumns` | 选择/排除列 |
| `Identity` | 恒等变换 |

---

## 窗口上下文

val 和 test 拆分在构建窗口时，会自动拼接前一个 split 尾部的数据作为上下文。

上下文长度 = `lookback + horizon - 1`，确保首步预测覆盖全部测试点：

```
训练 136 行 + 验证 29 行
                │ 上下文 14 行（来自训练集尾部）
                ▼
验证窗口: 41 行 → 41-12-3+1 = 27 个窗口 → 首步覆盖全部 29 点 ✅
```

---

## 扩展：添加新 Transform

```python
from EpiAI.dataset.base import Transform

class MyTransform(Transform):
    """自定义变换。"""
    def __init__(self, columns=None):
        super().__init__(columns=columns)

    def fit(self, df):
        # 在训练集上拟合
        self.mean_ = df[self.columns].mean()
        return self

    def transform(self, df):
        # 应用到所有 split
        result = df.copy()
        result[self.columns] = df[self.columns] - self.mean_
        return result

    def inverse(self, df):
        # 逆变换（预测结果反标准化用）
        result = df.copy()
        result[self.columns] = df[self.columns] + self.mean_
        return result
```

## 扩展：添加新 Split 策略

```python
from EpiAI.dataset.base import SplitStrategy
from EpiAI.dataset.container import SplitResult

class MySplit(SplitStrategy):
    def split(self, data: TimeSeriesData) -> SplitResult:
        # 返回三个 DataFrame：train / val / test
        return SplitResult(
            train_df=data.df.iloc[:100],
            val_df=data.df.iloc[100:130],
            test_df=data.df.iloc[130:],
        )
```
