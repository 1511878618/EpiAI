# EpiAI 数据管道重构文档

> **版本:** v2 (重构完成)
> **分支:** `main`
> **状态:** 稳定

---

## 目录

1. [为什么重构](#1-为什么重构)
2. [整体架构](#2-整体架构)
3. [快速入门](#3-快速入门)
4. [组件详解](#4-组件详解)
   - [4.1 数据容器 TimeSeriesData](#41-数据容器-timeseriesdata)
   - [4.2 数据加载器 DataLoader](#42-数据加载器-dataloader)
   - [4.3 拆分策略 SplitStrategy](#43-拆分策略-splitstrategy)
   - [4.4 数据变换 Transform](#44-数据变换-transform)
   - [4.5 滑动窗口 SlidingWindow](#45-滑动窗口-slidingwindow)
   - [4.6 编排器 ForecastPipeline](#46-编排器-forecastpipeline)
5. [场景示例](#5-场景示例)
6. [扩展指南](#6-扩展指南)
7. [迁移说明](#7-迁移说明)
8. [后续规划](#8-后续规划)

---

## 1. 为什么重构

原有的数据管道存在三个核心问题：

| 问题 | 旧代码 | 新架构 |
|------|--------|--------|
| **耦合过紧** | 归一化、拆分、滑窗都写死在 `builder.py` 的一个方法里 | 每个阶段独立为可插拔组件，自由组合 |
| **扩展困难** | 加一种拆分方式要改 builder 核心逻辑 | 新增一个 `SplitStrategy` 子类即可 |
| **不兼容** | `dataset/` 面向 `.pt` 3D tensor，`SimpleForecastDataModule` 面向 DataFrame，两套不互通 | 统一 `TimeSeriesData` 容器，格式无关 |

新架构的目标：

- **单一数据容器** — 无论数据来自 CSV、Feather 还是 PyTorch Tensor，最终都归一为 `TimeSeriesData`
- **策略模式** — 每个可变的步骤（拆分方式、变换方式）都定义为接口，自由组合
- **零 torch 依赖** — 数据预处理层只需 pandas + numpy，降低环境门槛
- **可逆变换** — 所有标量变换支持 `inverse()`，方便评估时还原预测值

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        ForecastPipeline                         │
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│   │ DataLoader│→│ Split    │→│ Transform │→│ SlidingWindow │  │
│   │          │  │ Strategy │  │ Pipeline │  │              │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│        ↓              ↓             ↓               ↓          │
│   TimeSeriesData  SplitResult  DataFrame →     WindowArrays    │
│                                  DataFrame     (3D numpy)      │
└─────────────────────────────────────────────────────────────────┘
                                       ↓
                                 PipelineBundle
                           (train_x/y, val_x/y, test_x/y)
```

**数据流：**
```
文件 (CSV/Feather/.pt)
  → TimeSeriesData (统一容器，含时间列、目标列、特征列、实体列)
    → SplitResult (train/val/test 的行索引)
      → 变换后的 DataFrame (每份拆分独立变换)
        → WindowArrays (3D numpy: windows × lookback × features)
          → PipelineBundle (最终产出)
```

所有组件在 `EpiAI.dataset` 下直接可导入：

```python
from EpiAI.dataset import (
    # 容器
    TimeSeriesData, SplitResult, WindowBundle,
    # 加载器
    CsvLoader, FeatherLoader, TensorLoader, load_data,
    # 拆分策略
    TimeSplit, EntitySplit, EntityTimeSplit,
    CustomIndexSplit, NoSplit, CrossValidationSplit,
    # 变换
    StandardScaler, RobustScaler, Log1pTransform, BoxCoxTransform,
    Identity, SelectColumns, DateFeatures, FeatureLag,
    # 滑窗 & 编排
    SlidingWindow, WindowArrays,
    ForecastPipeline, PipelineBundle,
    # 基础抽象
    DataLoader, SplitStrategy, Transform, Compose,
)
```

---

## 3. 快速入门

### 3.1 最简单的方式 — 一键快速管道

```python
from EpiAI.dataset import ForecastPipeline

bundle = ForecastPipeline.quick(
    path="data.csv",          # CSV 文件路径
    time_col="time",           # 时间列名
    target_cols="dengue",      # 目标变量（预测对象）
    feature_cols=["temp", "humid"],  # 输入特征
    lookback=12,               # 用过去 12 步预测未来
    horizon=3,                 # 预测未来 3 步
    train_ratio=0.7,           # 70% 训练
    val_ratio=0.15,            # 15% 验证
    normalize=True,            # 自动标准化
)

# 获取训练数据
X_train = bundle.train_x   # shape: (n_windows, 12, 2)
y_train = bundle.train_y   # shape: (n_windows, 3, 1)
X_val   = bundle.val_x
y_val   = bundle.val_y
X_test  = bundle.test_x
y_test  = bundle.test_y
```

### 3.2 完整控制 — 手动构建

```python
from EpiAI.dataset import *

pipeline = ForecastPipeline(
    loader=CsvLoader(
        time_col="time",
        target_cols=["dengue", "flu"],
        feature_cols=["temp", "humid"],
        entity_col="city",        # 多城市数据
    ),
    split=TimeSplit(train_ratio=0.6, val_ratio=0.2),
    transforms=Compose([
        Log1pTransform(columns=["dengue", "flu"]),       # 对数变换
        StandardScaler(columns=["temp", "humid"]),        # 标准化
        DateFeatures(time_col="time", features=["month", "season"]),
    ]),
    window=SlidingWindow(lookback=12, horizon=3, stride=1),
)

bundle = pipeline.run("data.csv")

print(f"训练样本: {bundle.n_train}")
print(f"特征维度: {bundle.n_features}")
print(f"预测目标: {bundle.target_names}")
```

### 3.3 仅做变换，不做滑窗

如果只需要变换后的 DataFrame（不做滑窗），将 `window=None`：

```python
pipeline = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="dengue",
                     feature_cols=["temp"]),
    split=TimeSplit(train_ratio=0.6, val_ratio=0.2),
    transforms=Compose([
        Log1pTransform(columns=["dengue"]),
        StandardScaler(),
    ]),
    window=None,  # 跳过滑窗，返回 DataFrames
)
bundle = pipeline.run("data.csv")
# bundle.train_x shape: (n_train, 1, n_features) ← dummy 第三维
```

---

## 4. 组件详解

### 4.1 数据容器 `TimeSeriesData`

所有数据加载器的统一返回类型。

```python
from EpiAI.dataset import TimeSeriesData

data = TimeSeriesData(
    df=df,                    # pd.DataFrame
    time_col="time",          # 时间列名
    target_cols=["dengue"],   # 目标列
    feature_cols=["temp"],    # 特征列
    entity_col="city",        # 可选：实体列（城市/地区）
)

# 自动推导的属性
data.n_entities     # 实体数量（无 entity_col 则为 1）
data.entity_values  # 实体列表，如 ["北京", "上海"]
data.time_range     # 时间范围 (min, max)
```

### 4.2 数据加载器 `DataLoader`

| 加载器 | 文件格式 | 用法 |
|--------|----------|------|
| `CsvLoader` | `.csv` | 最常见，需指定列映射 |
| `FeatherLoader` | `.feather` | 快速 IO，需 `pyarrow` |
| `TensorLoader` | `.pt` | 兼容旧版 3D tensor 格式 |

```python
# CSV（单城市）
loader = CsvLoader(
    time_col="date",
    target_cols="cases",
    feature_cols=["temp", "humidity", "rainfall"],
)

# CSV（多城市）
loader = CsvLoader(
    time_col="date",
    target_cols="cases",
    feature_cols=["temp"],
    entity_col="province",
)

# Tensor（旧版 3D 格式: [time, city, feature]）
loader = TensorLoader(
    target_feature_names=["登革热", "流感"],
    input_feature_mode="exclude_targets",
    province_label="province",
)

# 自动识别扩展名
data = load_data("data.csv",
    time_col="time", target_cols="dengue", feature_cols=["temp"])
```

### 4.3 拆分策略 `SplitStrategy`

#### `TimeSplit` — 按时间切分

两种模式：

```python
# 模式 A：按日期
split = TimeSplit(train_end="2020-12-31", val_end="2022-06-30")

# 模式 B：按比例（推荐）
split = TimeSplit(train_ratio=0.7, val_ratio=0.15)  # 70% train, 15% val, 15% test
```

#### `EntitySplit` — 按实体（城市）切分

```python
split = EntitySplit(
    train_entities=["北京", "上海", "广州"],
    val_entities=["深圳"],
    test_entities=["杭州"],
)
```

#### `EntityTimeSplit` — 每个实体独立时间窗口

```python
split = EntityTimeSplit({
    "北京": ("2018-01-01", "2020-06-01"),
    "上海": ("2019-01-01", "2020-12-01"),
    "广州": ("2017-01-01", "2021-01-01"),
})
```

#### `CustomIndexSplit` — 自定义索引

```python
split = CustomIndexSplit(
    train_idx=[0, 1, 2, 3, 4, 5, 6, 7],
    val_idx=[8, 9, 10],
    test_idx=[11, 12, 13, 14],
)
```

#### 其他

```python
split = NoSplit()        # 全部作为训练
split = CrossValidationSplit(n_splits=5, val_horizon=12)  # 时序 CV
```

### 4.4 数据变换 `Transform`

所有变换都实现了 `transform()`，需要时可选的 `fit()` 和 `inverse()`。

```python
# 组合多个变换
pipeline = Compose([
    Log1pTransform(columns=["dengue", "flu"]),   # 对数变换
    StandardScaler(columns=["dengue", "temp"]),   # Z-score 标准化
    DateFeatures(time_col="time",
                 features=["month", "season"]),   # 时间特征提取
    FeatureLag(columns=["temp"], lags=[1, 3, 6]), # 滞后特征
])
pipeline.fit(train_df)             # 在训练集上拟合
transformed = pipeline.transform(val_df)   # 应用到验证集
restored = pipeline.inverse(transformed)   # 逆向还原（评估用）
```

| 变换 | 说明 | 可逆 |
|------|------|------|
| `Identity` | 不做任何变换 | ✅ |
| `StandardScaler` | Z-score 标准化：(x-μ)/σ | ✅ |
| `RobustScaler` | 鲁棒标准化：(x-median)/IQR | ✅ |
| `Log1pTransform` | log(1+x) 变换 | ✅ |
| `BoxCoxTransform` | Box-Cox 幂变换（需 scipy） | ✅ |
| `SelectColumns` | 选列或删列 | ❌ |
| `DateFeatures` | 从时间列提取 year/month/dayofweek/season | ❌ |
| `FeatureLag` | 添加滞后特征列 | ❌ |

### 4.5 滑动窗口 `SlidingWindow`

不是 `Transform` 子类（因为它改变了数据维度），是在所有变换完成后应用。

```python
sw = SlidingWindow(lookback=12, horizon=3, ahead=0, stride=1)
windows = sw.apply(
    df=processed_df,           # 已变换的 DataFrame
    target_cols=["dengue"],    # 目标列
    feature_cols=["temp"],     # 特征列
    entity_col="city",         # 可选，多实体时会按实体分别滑窗
)
# windows.x  shape: (n_samples, 12, 1)  特征窗口
# windows.y  shape: (n_samples, 3, 1)   预测目标
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lookback` | 必填 | 用过去 N 个时间步预测未来 |
| `horizon` | 必填 | 预测未来 M 个时间步 |
| `ahead` | 0 | 输入与输出之间的间隔 |
| `stride` | 1 | 滑窗步长（>1 可减少样本量） |

### 4.6 编排器 `ForecastPipeline`

把加载、拆分、变换、滑窗串联起来的入口。

| 方法 | 说明 |
|------|------|
| `__init__(loader, split, transforms, window)` | 构建管道 |
| `run(path)` | 执行完整流程，返回 `PipelineBundle` |
| `quick(path, ...)` | 快速模式：一行完成 |

`PipelineBundle` 属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `train_x`, `train_y` | np.ndarray | 训练集窗口数组 |
| `val_x`, `val_y` | np.ndarray | 验证集窗口数组 |
| `test_x`, `test_y` | np.ndarray | 测试集窗口数组 |
| `feature_names` | list[str] | 特征名列表 |
| `target_names` | list[str] | 目标名列表 |
| `n_train`, `n_val`, `n_test` | int | 各集样本数 |
| `lookback`, `horizon` | int | 窗口配置 |
| `n_features`, `n_targets` | int | 维度 |
| `transforms` | Compose or None | 变换管道（可用于 inverse） |
| `data` | TimeSeriesData | 原始数据引用 |

---

## 5. 场景示例

### 场景 A：单城市登革热预测

```python
from EpiAI.dataset import ForecastPipeline

bundle = ForecastPipeline.quick(
    path="data.csv",
    time_col="Year/Month",
    target_cols="Case number",
    feature_cols=["Case number"],  # 自回归：用历史病例预测未来
    lookback=12,    # 用过去一年
    horizon=3,      # 预测未来一个季度
)
```

### 场景 B：多城市 + 多疾病，按城市和时间组合拆分

```python
from EpiAI.dataset import *

pipeline = ForecastPipeline(
    loader=CsvLoader(
        time_col="Year/Month",
        target_cols=["登革热", "流感"],
        feature_cols=["登革热", "流感", "temp", "humid"],
        entity_col="city",
    ),
    split=EntityTimeSplit({
        "北京": ("2018-01", "2020-06"),
        "上海": ("2019-01", "2020-12"),
        "广州": ("2017-06", "2020-09"),
    }),
    transforms=Compose([
        Log1pTransform(columns=["登革热", "流感"]),
        StandardScaler(),
        DateFeatures(time_col="Year/Month",
                     features=["month", "quarter", "season"]),
    ]),
    window=SlidingWindow(lookback=12, horizon=3),
)
bundle = pipeline.run("china_disease_data.csv")
```

### 场景 C：使用旧版 .pt tensor 数据

```python
from EpiAI.dataset import *

pipeline = ForecastPipeline(
    loader=TensorLoader(
        target_feature_names=["登革热", "流感"],
        input_feature_mode="exclude_targets",
        mark_feature_names=["年", "月"],
    ),
    split=TimeSplit(train_ratio=0.6, val_ratio=0.2),
    transforms=Compose([
        Log1pTransform(),
        StandardScaler(),
    ]),
    window=SlidingWindow(lookback=24, horizon=6),
)
bundle = pipeline.run("data/Align_data_tensor_with_name.pt")
```

### 场景 D：自定义拆分 + 无滑窗（仅做变换）

```python
pipeline = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols=["feature1", "feature2"]),
    split=CustomIndexSplit(
        train_idx=range(0, 80),
        val_idx=range(80, 100),
        test_idx=range(100, 120),
    ),
    transforms=Compose([
        RobustScaler(),
        FeatureLag(columns=["cases"], lags=[1, 2, 3]),
    ]),
    window=None,  # 不做滑窗
)
bundle = pipeline.run("data.csv")
```

---

## 6. 扩展指南

### 添加新的数据加载器

继承 `DataLoader`，实现 `load()` 方法返回 `TimeSeriesData`：

```python
from EpiAI.dataset import DataLoader, TimeSeriesData

class ParquetLoader(DataLoader):
    def __init__(self, time_col, target_cols, feature_cols, entity_col=None):
        self.time_col = time_col
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.entity_col = entity_col

    def load(self, path):
        df = pd.read_parquet(path)
        return TimeSeriesData(
            df=df,
            time_col=self.time_col,
            target_cols=self.target_cols,
            feature_cols=self.feature_cols,
            entity_col=self.entity_col,
            metadata={"loader": "parquet", "path": path},
        )
```

然后注册到自动发现系统：

```python
from EpiAI.dataset.loaders import _register
_register(".parquet", ParquetLoader)
# 现在 load_data("data.parquet", ...) 会自动使用 ParquetLoader
```

### 添加新的拆分策略

继承 `SplitStrategy`，实现 `split()` 返回 `SplitResult`：

```python
from EpiAI.dataset import SplitStrategy, SplitResult

class StratifiedTimeSplit(SplitStrategy):
    """时间分层采样，保证各时段在每个 split 中都有代表。"""
    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def split(self, data):
        n = len(data.df)
        indices = np.arange(n)
        # 自定义逻辑...
        return SplitResult(
            train_idx=indices[:train_end],
            val_idx=indices[train_end:val_end],
            test_idx=indices[val_end:],
        )
```

### 添加新的变换

继承 `Transform`，实现 `transform()`，可选 `fit()` 和 `inverse()`：

```python
from EPIai.dataset import Transform

class DifferenceTransform(Transform):
    """一阶差分：消除趋势。"""
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, df):
        df = df.copy()
        cols = self.columns or df.select_dtypes(include=[np.number]).columns
        df[cols] = df[cols].diff().fillna(0)
        return df
```

### 扩展指南的核心原则

1. **单一职责** — 每个组件只做一件事
2. **接口一致** — `fit()` 在训练数据上学习参数，`transform()` 应用到任何数据
3. **可逆性** — 对目标变量的变换尽量实现 `inverse()`
4. **无状态返回** — 永远不修改输入的 DataFrame，返回副本

---

## 7. 迁移说明

### 旧代码如何迁移

原有代码（使用 `DatasetConfig` + `MultiTargetCityDatasetBuilder`）**仍然可以工作**，旧接口完整保留。两者可以共存：

```python
# 旧代码 — 依然可用
from EpiAI.dataset import DatasetConfig, MultiTargetCityDatasetBuilder
config = DatasetConfig(data_path="...", target_feature_names=[...])
bundle = MultiTargetCityDatasetBuilder(config).build()

# 新代码 — 推荐
from EpiAI.dataset import ForecastPipeline
bundle = ForecastPipeline.quick(path="...", ...)
```

### 迁移对照表

| 旧功能 | 新功能 |
|--------|--------|
| `DatasetConfig` | `ForecastPipeline` 的参数 |
| `MultiTargetCityDatasetBuilder` | `ForecastPipeline.run()` |
| `DiseaseTensorData` | `TimeSeriesData` |
| `DatasetBundle` (含 train_input/train_target) | `PipelineBundle` (含 train_x/train_y) |
| `CitySplitter` | `EntitySplit` |
| `TensorStandardScaler` | `StandardScaler` |
| `make_sliding_windows()` | `SlidingWindow.apply()` |
| `load_disease_tensor()` | `TensorLoader().load()` |

---

## 8. 后续规划

本次更新仅涉及**数据管道层**。后续计划逐步重构：

| 阶段 | 模块 | 内容 |
|------|------|------|
| ✅ **已完成** | `dataset/` | 数据加载、拆分、变换、滑窗、编排 |
| ⬜ **待更新** | `models/` | 统一模型接口，支持训练/预测/评估通用协议 |
| ⬜ **待更新** | `losses/` | 保持现状（已设计良好），考虑与 Transform 的 inverse 联动 |
| ⬜ **待更新** | `train.py` | 通用训练循环，支持 Lightning / 自定义循环 |
| ⬜ **待更新** | `evaluation/` | 统一评估指标和可视化 |

---

> **文档最后更新:** 2026-06-13
> **分支:** `refactor/data-pipeline`
> **测试覆盖:** 54 个测试用例，全部通过
