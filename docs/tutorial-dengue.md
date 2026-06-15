# EpiAI 使用教程：全国登革热月发病数预测

> 纯自回归（用历史病例预测未来病例），对比所有可用模型。

---

## 1. 环境准备

```python
import sys, os
sys.path.insert(0, os.path.abspath("../src"))
import numpy as np
import pandas as pd

from EpiAI.models import sklearn_models, ts_models
from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, Compose,
    StandardScaler, SlidingWindow,
)
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import InferencePipeline
```

---

## 2. 数据加载

```python
df_raw = pd.read_csv("../data/Infective_disease_china-V3.csv")

# 筛选登革热
df = df_raw[df_raw["Diseases"] == "登革热 Dengue fever"].copy()
df = df.rename(columns={"Year/Month": "time", "Case number": "cases"})
df = df[["time", "cases"]].reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"])
df["cases"] = df["cases"].astype(float)

print(f"登革热: {len(df)} 个月 ({df['time'].min().date()} ~ {df['time'].max().date()})")
print(f"病例数范围: {df['cases'].min():.0f} ~ {df['cases'].max():.0f}")
```

---

## 3. 数据管道

纯自回归：用历史 `cases` 预测未来 `cases`，不做任何特征工程。

```python
df.to_csv("/tmp/dengue.csv", index=False)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols="cases"),    # 自回归
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,                           # 无变换，纯原始值
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/dengue.csv")

print(f"训练: {bundle.train_x.shape}  ({bundle.n_train} 个窗口)")
print(f"验证: {bundle.val_x.shape}     ({bundle.n_val} 个窗口)")
print(f"测试: {bundle.test_x.shape}     ({bundle.n_test} 个窗口)")
print(f"每个窗口: {bundle.lookback} 个月历史 → 预测 {bundle.horizon} 个月")
```

---

## 4. 模型训练与对比

遍历所有可用模型，自动跳过因缺少依赖而无法导入的模型。

### 4.1 窗口模型（Sklearn 族）

```python
results = []
sklearn_names = ["RF", "XGB", "LGBM", "SVR", "GLM"]

for name in sklearn_names:
    try:
        model_cls = get(name)
    except KeyError:
        print(f"  ⚠️ {name}: 未注册，跳过")
        continue

    # 构造参数
    extra = {}
    if name == "RF":
        extra["rf_params"] = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    elif name == "XGB":
        extra["xgb_params"] = {"n_estimators": 200, "random_state": 42}
    elif name == "LGBM":
        extra["lgbm_params"] = {"n_estimators": 200, "random_state": 42}
    elif name == "SVR":
        extra["svm_params"] = {"kernel": "rbf", "C": 1.0}

    try:
        if name in ["ETS", "ARIMA"]:
            # TS models have their own constructor — don't pass window params
            model = model_cls(**extra)
        else:
            model = model_cls(
                input_dim=bundle.n_features, lookback=12,
                horizon=3, target_dim=1, **extra,
            )
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        metric = result.metrics.iloc[0]
        results.append({"model": name, "MAE": metric["MAE"],
                        "RMSE": metric["RMSE"], "R2": metric["R2"]})
        print(f"  ✅ {name:5s}  MAE={metric['MAE']:.1f}  R²={metric['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:5s}  {e}")
```

### 4.2 时序模型（TimeSeries 族）

时序模型用纯未来预测（`forecast`），不走窗口数据。

```python
ts_names = ["ETS", "ARIMA"]

for name in ts_names:
    try:
        model_cls = get(name)
    except KeyError:
        print(f"  ⚠️ {name}: 未注册，跳过")
        continue

    try:
        if name == "ETS":
            model = model_cls(seasonal_periods=12, seasonal="add", trend="add")
        elif name == "ARIMA":
            model = model_cls(seasonal=True, m=12)

        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        metric = result.metrics.iloc[0]
        results.append({"model": name, "MAE": metric["MAE"],
                        "RMSE": metric["RMSE"], "R2": metric["R2"]})
        print(f"  ✅ {name:5s}  MAE={metric['MAE']:.1f}  R²={metric['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:5s}  {e}")
```

### 4.3 结果汇总

```python
comp = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n模型对比（按 R² 排序）：")
print(comp.to_string(index=False))
```

---

## 5. 最佳模型推理部署

```python
best_name = comp.iloc[0]["model"]
print(f"最佳模型: {best_name}")

# 重新训练最佳模型
best_extra = {}
if best_name == "RF":
    best_extra["rf_params"] = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
elif best_name == "XGB":
    best_extra["xgb_params"] = {"n_estimators": 200, "random_state": 42}
elif best_name == "ETS":
    best_extra = {"seasonal_periods": 12, "seasonal": "add", "trend": "add"}
elif best_name == "ARIMA":
    best_extra = {"seasonal": True, "m": 12}

best_cls = get(best_name)
if best_name in ["ETS", "ARIMA"]:
    best_model = best_cls(**best_extra)
else:
    best_model = best_cls(
        input_dim=bundle.n_features, lookback=12,
        horizon=3, target_dim=1, **best_extra,
    )
best_result = EpiAITrainer(model=best_model, verbose=False).fit(bundle)
best_result._bundle = bundle

# 部署为推理管道
inferer = InferencePipeline.from_train_result(best_result)
print(f"最佳模型: {best_name} → {inferer}")

# 用最后 15 个月数据预测未来 3 个月
new_data = bundle.train_df.tail(15).copy()[bundle.feature_names]
pred = inferer.predict(new_data)
print(f"\n预测未来 3 个月登革热病例: {pred[0, :, 0].round(0).astype(int)}")

# 保存
inferer.save("/tmp/dengue_best_model.zip")
print("\n模型已保存到 /tmp/dengue_best_model.zip")
```

---

## 6. 附录：可用模型

```python
print("Torch 模型:", list_models("torch"))
print("Sklearn 模型:", list_models("sklearn"))
print("TimeSeries 模型:", list_models("ts"))
```

依赖安装：

```bash
# 基础
pip install -e .

# sklearn 模型
pip install -e ".[xgb,lgbm]"

# 全部
pip install -e ".[all]"
```
