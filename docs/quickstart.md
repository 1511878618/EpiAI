# 快速上手

5 分钟跑通 EpiAI：加载数据、训练模型、查看结果。

---

## 安装

```bash
pip install -e .                     # 基础版（numpy/pandas/sklearn）
pip install -e ".[torch]"            # + 深度学习模型
pip install -e ".[all]"             # 全部模型
```

## 数据

框架内置了登革热月发病数样例数据（2010–2026）。

```python
import pandas as pd

df = pd.read_csv("data/Infective_disease_china-V3.csv")
df = df[df["Diseases"] == "登革热 Dengue fever"]
print(f"样本数: {len(df)}, 时间: {df['Year/Month'].min()} ~ {df['Year/Month'].max()}")
```

## 数据管道

```python
from EpiAI.dataset import ForecastPipeline, CsvLoader, TimeSplit, SlidingWindow

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="Year/Month", target_cols="Case number",
                     feature_cols="Case number"),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,
    window=SlidingWindow(lookback=12, horizon=3),
).run("data/Infective_disease_china-V3.csv")

print(f"训练窗口: {bundle.train_x.shape}")
print(f"测试窗口: {bundle.test_x.shape}")
```

## 训练模型

```python
from EpiAI.models.registry import get
from EpiAI.trainer import EpiAITrainer

model = get("RF")(input_dim=bundle.n_features, lookback=12,
                  horizon=3, target_dim=1)
result = EpiAITrainer(model=model, verbose=False).fit(bundle)

print(result.metrics)
```

输出示例：

```
         paradigm       MAE       RMSE       R2
model
RF         sklearn   758.97   1366.75   0.693
```

## 改变模型

只需替换 `get()` 中的名字：

```python
# 深度学习
result = EpiAITrainer(model=get("LSTM")(...), verbose=False).fit(bundle)

# 时间序列（无需滑窗参数）
result = EpiAITrainer(model=get("ETS")(seasonal_periods=12), verbose=False).fit(bundle)
```

## 下一步

完整的端到端教程见 [`tutorial/tutorial-full.ipynb`](../tutorial/tutorial-full.ipynb)。
模块详细说明见 [数据模块指南](guides/dataset.md)、[模型模块指南](guides/models.md)、
[训练模块指南](guides/training.md)、[部署模块指南](guides/deployment.md)。
