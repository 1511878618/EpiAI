# EpiAI 使用教程：全国登革热月发病数预测

> 本教程使用 `Infective_disease_china-V3.csv` 中的全国登革热数据，
> 完整演示 EpiAI 从数据加载到模型推理的全流程。

---

## 1. 环境准备

```python
import sys, os
sys.path.insert(0, "src")          # 确保能找到 EpiAI 包
import numpy as np
import pandas as pd

# 触发模型注册
from EpiAI.models import sklearn_models, ts_models

from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, Compose,
    Log1pTransform, StandardScaler, DateFeatures,
    FeatureLag, SlidingWindow,
)
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import InferencePipeline
```

---

## 2. 数据加载与探索

```python
df_raw = pd.read_csv("data/Infective_disease_china-V3.csv")
print(f"总行数: {len(df_raw)}, 疾病种类: {df_raw['Diseases'].nunique()}")
print(f"时间范围: {df_raw['Year/Month'].min()} ~ {df_raw['Year/Month'].max()}")

# 筛选登革热
df = df_raw[df_raw["Diseases"] == "登革热 Dengue fever"].copy()
df = df.rename(columns={"Year/Month": "time", "Case number": "cases"})
df = df[["time", "cases"]].reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"])
df["cases"] = df["cases"].astype(float)

print(f"\n登革热数据: {len(df)} 个月 ({df['time'].min().date()} ~ {df['time'].max().date()})")
print(df.head())
print(f"\n统计:\n{df['cases'].describe()}")
```

---

## 3. 数据管道

将病例数转化为自回归预测任务：用历史病例预测未来病例。
由于 CSV 是长格式（多疾病混合），先筛选登革热并保存临时文件：

```python
tmp_csv = "/tmp/dengue_national.csv"
df.to_csv(tmp_csv, index=False)

pipeline = ForecastPipeline(
    loader=CsvLoader(
        time_col="time",
        target_cols="cases",
        feature_cols="cases",         # 自回归：用历史病例预测未来
    ),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=Compose([
        Log1pTransform(columns=["cases"]),          # 对数变换（病例数偏态）
        StandardScaler(columns=["cases"]),          # 标准化
        DateFeatures(time_col="time", features=[
            "month", "season",
        ]),                                        # 添加月份和季节特征
        FeatureLag(columns=["cases"],
                   lags=[1, 2, 3, 6, 12]),          # 滞后特征
    ]),
    window=SlidingWindow(lookback=12, horizon=3),   # 用过去一年预测一季度
)

bundle = pipeline.run(tmp_csv)

print(f"特征 ({len(bundle.feature_names)}): {bundle.feature_names}")
print(f"目标: {bundle.target_names}")
print(f"训练窗口: {bundle.train_x.shape}")
print(f"验证窗口: {bundle.val_x.shape}")
print(f"测试窗口: {bundle.test_x.shape}")
```

**输出示例：**
```
特征 (8): ['cases', 'month', 'season', 'cases_lag_1',
           'cases_lag_2', 'cases_lag_3', 'cases_lag_6', 'cases_lag_12']
目标: ['cases']
训练窗口: (107, 12, 8)
验证窗口: (11, 12, 8)
测试窗口: (12, 12, 8)
```

> 注意：因为一个特征为 `cases`（本身），`Log1pTransform` 和
> `StandardScaler` 已在训练集上拟合，变换会应用到所有 split。

---

## 4. 训练模型

### 4.1 随机森林（Sklearn 族）

```python
model_rf = get("RF")(
    input_dim=bundle.n_features,
    lookback=12,
    horizon=3,
    target_dim=1,
    rf_params={"n_estimators": 200, "max_depth": 10, "random_state": 42},
)

result_rf = EpiAITrainer(model=model_rf, verbose=False).fit(bundle)

print("\n随机森林结果:")
print(result_rf.metrics.to_string(index=False))
```

### 4.2 XGBoost（Sklearn 族）

```python
model_xgb = get("XGB")(
    input_dim=bundle.n_features,
    lookback=12,
    horizon=3,
    target_dim=1,
    xgb_params={"n_estimators": 200, "learning_rate": 0.05,
                "max_depth": 5, "random_state": 42},
)

result_xgb = EpiAITrainer(model=model_xgb, verbose=False).fit(bundle)

print("\nXGBoost 结果:")
print(result_xgb.metrics.to_string(index=False))
```

### 4.3 ETS（TimeSeries 族）

ETS 模型不需要窗口数据，直接从 bundle 提取原始序列：

```python
model_ets = get("ETS")(
    seasonal_periods=12,   # 年度季节性周期
    seasonal="add",
    trend="add",
)

result_ets = EpiAITrainer(model=model_ets, verbose=False).fit(bundle)

print("\nETS 结果:")
print(result_ets.metrics.to_string(index=False))
```

---

## 5. 模型对比

```python
results = {"随机森林 RF": result_rf, "XGBoost": result_xgb, "ETS": result_ets}

comp_df = pd.concat([
    r.metrics.assign(model=name) for name, r in results.items()
], ignore_index=True)

pivot = comp_df.pivot_table(
    index="model", values=["MAE", "RMSE", "R2", "PearsonR"], aggfunc="first",
)
print("\n模型对比:")
print(pivot.round(3).to_string())
```

---

## 6. 推理：对新数据做预测

选最优模型部署为推理管道：

```python
best_result = result_xgb    # 假设 XGBoost 表现最好
best_result._bundle = bundle

inferer = InferencePipeline.from_train_result(best_result)
print(f"推理管道: {inferer}")

# 取最后 15 行作为"新数据"（需要至少 lookback+horizon=15 行）
new_data = bundle.train_df.tail(15).copy()[bundle.feature_names]
print(f"输入: {len(new_data)} 行 × {len(new_data.columns)} 特征")

pred = inferer.predict(new_data)
print(f"\n预测登革热未来 3 个月: {pred[0, :, 0].round(0).astype(int)}")
```

### 保存与加载

```python
# 保存
inferer.save("/tmp/dengue_xgb_model.zip")
print("✅ 已保存到 /tmp/dengue_xgb_model.zip")

# 加载
loaded = InferencePipeline.load("/tmp/dengue_xgb_model.zip")
pred2 = loaded.predict(new_data)
assert np.allclose(pred, pred2, atol=1e-5), "保存/加载后预测不一致"
print("✅ 加载验证通过")
```

---

## 7. ETS 纯未来预测

```python
result_ets._bundle = bundle
inferer_ets = InferencePipeline.from_train_result(result_ets)

forecast = inferer_ets.forecast(12)
print(f"未来 12 个月登革热预测值:")
print(forecast.ravel().round(0).astype(int))
```

---

## 8. 完整流程函数

```python
def run_dengue_forecast(csv_path="data/Infective_disease_china-V3.csv"):
    """全国登革热预测完整流程。"""

    # 1. 加载 + 筛选
    df = pd.read_csv(csv_path)
    df = df[df["Diseases"] == "登革热 Dengue fever"].copy()
    df = df.rename(columns={"Year/Month": "time", "Case number": "cases"})
    df = df[["time", "cases"]].reset_index(drop=True)
    df["cases"] = df["cases"].astype(float)
    df.to_csv("/tmp/dengue.csv", index=False)

    # 2. 数据管道
    bundle = ForecastPipeline(
        loader=CsvLoader(time_col="time", target_cols="cases",
                         feature_cols="cases"),
        split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
        transforms=Compose([
            Log1pTransform(columns=["cases"]),
            StandardScaler(columns=["cases"]),
            DateFeatures(time_col="time", features=["month", "season"]),
            FeatureLag(columns=["cases"], lags=[1, 2, 3, 6, 12]),
        ]),
        window=SlidingWindow(lookback=12, horizon=3),
    ).run("/tmp/dengue.csv")

    # 3. 训练 XGBoost
    model = get("XGB")(
        input_dim=bundle.n_features, lookback=12,
        horizon=3, target_dim=1,
        xgb_params={"n_estimators": 200, "random_state": 42},
    )
    result = EpiAITrainer(model=model, verbose=False).fit(bundle)

    # 4. 评估
    print(result.metrics.to_string(index=False))

    # 5. 部署
    result._bundle = bundle
    inferer = InferencePipeline.from_train_result(result)
    inferer.save("/tmp/dengue_xgb.zip")

    return result, inferer

# result, inferer = run_dengue_forecast()
```

---

## 附录

### 可用模型一览

```python
print("Torch 模型:", list_models("torch"))
print("Sklearn 模型:", list_models("sklearn"))
print("TimeSeries 模型:", list_models("ts"))
```

### 依赖安装

```bash
# 最小安装（数据管道基础功能）
pip install -e .

# 加 sklearn 模型（推荐）
pip install -e ".[xgb,lgbm]"

# 全部（含 torch）
pip install -e ".[all]"
```

### 快速切换疾病

只需改一行筛选条件即可换疾病：

```python
df = df_raw[df_raw["Diseases"] == "流行性感冒 Influenza"].copy()
#                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

也可换多城市（在 CsvLoader 中设置 `entity_col="province"`）。
