# 数据模块指南

## 功能介绍

数据模块负责将原始数据转换为模型可用的格式，位于 `EpiAI.dataset`。

### 关键类

| 类 | 作用 |
|----|------|
| `CsvLoader` / `FeatherLoader` / `TensorLoader` | 从文件加载数据 |
| `TimeSplit` / `EntitySplit` / `EntityTimeSplit` | 拆分数据 |
| `StandardScaler` / `RobustScaler` / `Log1pTransform` / `DateFeatures` / `FeatureLag` | 数据变换 |
| `SlidingWindow` | 生成 3D 窗口数组 |
| `ForecastPipeline` | 编排完整流程 |
| `PipelineBundle` | 管道输出结果 |

### 使用示例

```python
from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, Compose,
    StandardScaler, SlidingWindow,
)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols=["cases", "temp"]),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=Compose([StandardScaler(columns=["cases", "temp"])]),
    window=SlidingWindow(lookback=12, horizon=3),
).run("/path/to/data.csv")

# bundle 包含:
#   train_x: (N_train, 12, 2)   窗口输入
#   train_y: (N_train, 3, 1)    窗口目标
#   val_x, val_y, test_x, test_y
#   get_y_series("test"): 原始测试序列（给 TS 模型用）
```

---

## 扩展指南

### 添加新 Transform

1. 继承 `EpiAI.dataset.base.Transform`
2. 实现 `fit()` / `transform()` / `inverse()` 三个方法
3. 放入 `src/EpiAI/dataset/transforms.py`

```python
from EpiAI.dataset.base import Transform

class DifferenceTransform(Transform):
    """一阶差分变换。"""
    def __init__(self, columns=None):
        super().__init__(columns=columns)

    def fit(self, df):
        # 拟合第一行值用于逆运算
        self.first_values_ = df[self.columns].iloc[0].to_dict()
        return self

    def transform(self, df):
        result = df.copy()
        result[self.columns] = df[self.columns].diff()
        return result

    def inverse(self, df):
        result = df.copy()
        for col in self.columns:
            if col in result.columns:
                # 累加恢复
                cumsum = result[col].cumsum()
                result[col] = self.first_values_[col] + cumsum
        return result
```

无需修改其他文件，`Compose` 会自动组合。

### 添加新 Split 策略

1. 继承 `EpiAI.dataset.base.SplitStrategy`
2. 实现 `split(data: TimeSeriesData) -> SplitResult`
3. 放入 `src/EpiAI/dataset/splits.py`

```python
from EpiAI.dataset.base import SplitStrategy
from EpiAI.dataset.container import SplitResult, TimeSeriesData

class LastYearSplit(SplitStrategy):
    """以最后 12 个月为测试集。"""
    def __init__(self):
        super().__init__(train_size=None, val_size=None, test_size=None)

    def split(self, data: TimeSeriesData) -> SplitResult:
        df = data.df
        return SplitResult(
            train_df=df.iloc[:-12],
            val_df=df.iloc[-12:-6],
            test_df=df.iloc[-6:],
        )
```
