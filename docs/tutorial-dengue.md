# EpiAI 使用教程：全国登革热月发病数预测

> 纯自回归（用历史病例预测未来病例），完整演示数据加载 → 模型训练 → 可视化 → 部署

---

## 1. 环境准备

```python
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 注册所有模型
from EpiAI.models import sklearn_models, ts_models
from EpiAI.models import torch_models          # 需要 PyTorch，可选

from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit,
    SlidingWindow,
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

# 可视化原始序列
plt.figure(figsize=(14, 4))
plt.plot(df["time"], df["cases"], color="#2c3e50")
plt.title("全国登革热月发病数 (2010–2026)", fontsize=13)
plt.ylabel("病例数")
plt.grid(alpha=0.3)
plt.show()
```

---

## 3. 数据管道

用历史 12 个月预测未来 3 个月，纯自回归、无特征工程。

```python
df.to_csv("/tmp/dengue.csv", index=False)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols="cases"),        # 自回归
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,                               # 无变换
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/dengue.csv")

print(f"训练窗口: {bundle.train_x.shape}  ({bundle.n_train} 个)")
print(f"验证窗口: {bundle.val_x.shape}   ({bundle.n_val} 个)")
print(f"测试窗口: {bundle.test_x.shape}   ({bundle.n_test} 个)")
```

---

## 4. 深度学习模型（Torch 族）

需要安装 PyTorch：`pip install torch`。

```python
results = []

torch_names = ["MLP", "LSTM", "CNN", "ResNet", "TCN"]

for name in torch_names:
    try:
        model_cls = get(name)
    except KeyError:
        print(f"  ⚠️ {name}: 未注册")
        continue

    try:
        model = model_cls(
            input_dim=bundle.n_features,
            lookback=12, horizon=3, target_dim=1,
        )
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        m = result.metrics.iloc[0]
        results.append({"model": name, "MAE": m["MAE"],
                        "RMSE": m["RMSE"], "R2": m["R2"],
                        "PearsonR": m["PearsonR"]})
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}  r={m['PearsonR']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:10s}  {str(e)[:60]}")
```

---

## 5. 机器学习模型（Sklearn 族）

```python
sklearn_names = ["RF", "XGB", "LGBM", "SVR", "GLM"]

for name in sklearn_names:
    try:
        model_cls = get(name)
    except KeyError:
        print(f"  ⚠️ {name}: 未注册")
        continue

    extra = {}
    if name == "RF":   extra["rf_params"] = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    elif name == "XGB": extra["xgb_params"] = {"n_estimators": 200, "random_state": 42}
    elif name == "LGBM": extra["lgbm_params"] = {"n_estimators": 200, "random_state": 42}
    elif name == "SVR": extra["svm_params"] = {"kernel": "rbf", "C": 1.0}

    try:
        model = model_cls(
            input_dim=bundle.n_features, lookback=12,
            horizon=3, target_dim=1, **extra,
        )
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        m = result.metrics.iloc[0]
        results.append({"model": name, "MAE": m["MAE"],
                        "RMSE": m["RMSE"], "R2": m["R2"],
                        "PearsonR": m["PearsonR"]})
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}  r={m['PearsonR']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:10s}  {str(e)[:60]}")
```

---

## 6. 时间序列模型（TimeSeries 族）

TS 模型不走窗口数据，自动剥离与目标重叠的特征列。

```python
ts_names = ["ETS", "ARIMA"]

for name in ts_names:
    try:
        model_cls = get(name)
    except KeyError:
        print(f"  ⚠️ {name}: 未注册")
        continue

    try:
        extra = {"seasonal_periods": 12, "seasonal": "add", "trend": "add"} if name == "ETS" \
                else {"seasonal": True, "m": 12}
        model = model_cls(**extra)
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        m = result.metrics.iloc[0]
        results.append({"model": name, "MAE": m["MAE"],
                        "RMSE": m["RMSE"], "R2": m["R2"],
                        "PearsonR": m["PearsonR"]})
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}  r={m['PearsonR']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:10s}  {str(e)[:60]}")
```

---

## 7. 结果汇总

```python
comp = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n所有模型对比（按 R² 排序）：")
print(comp.to_string(index=False))
```

---

## 8. 最佳模型可视化

```python
best_name = comp.iloc[0]["model"]
print(f"最佳模型: {best_name}")

# 重训最佳模型（通用辅助函数）
def build_model(name, bundle):
    if name in ts_names:
        extra = {"seasonal_periods": 12, "seasonal": "add", "trend": "add"} if name == "ETS" \
                else {"seasonal": True, "m": 12}
        return get(name)(**extra)
    extra = {}
    if name == "RF":   extra["rf_params"] = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    elif name == "XGB": extra["xgb_params"] = {"n_estimators": 200, "random_state": 42}
    elif name == "LGBM": extra["lgbm_params"] = {"n_estimators": 200, "random_state": 42}
    elif name == "SVR": extra["svm_params"] = {"kernel": "rbf", "C": 1.0}
    return get(name)(input_dim=bundle.n_features, lookback=12, horizon=3, target_dim=1, **extra)

best_model = build_model(best_name, bundle)
best_result = EpiAITrainer(model=best_model, verbose=False).fit(bundle)
best_result._bundle = bundle

# 提取预测 vs 实际
y_test = bundle.get_y_series("test")
n = best_result.predictions.shape[0]
y_pred = best_result.predictions[:, -1, 0]
y_true = y_test[:n, 0]

# 绘图
plt.figure(figsize=(12, 5))
plt.plot(y_true, "o-", label="实际值", color="#2c3e50", alpha=0.8)
plt.plot(y_pred, "s--", label=f"预测值 ({best_name})", color="#e74c3c", alpha=0.7)
plt.xlabel("测试样本序号", fontsize=11)
plt.ylabel("登革热病例数", fontsize=11)
plt.title(f"{best_name} — 登革热月发病数预测", fontsize=13, fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

m = best_result.metrics.iloc[0]
txt = f"MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  R²={m['R2']:.3f}  r={m['PearsonR']:.3f}"
plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
         fontsize=11, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.tight_layout()
plt.savefig("/tmp/dengue_best_prediction.png", dpi=150)
plt.show()
```

---

## 9. 模型部署

不同模型族的推理方式不同，根据 best model 的 paradigm 自动路由：

```python
inferer = InferencePipeline.from_train_result(best_result)
print(f"部署: {inferer}")

paradigm = best_result.model.paradigm()

if paradigm == "ts":
    # 时序模型 → 纯未来预测
    forecast = inferer.forecast(steps=6)
    print(f"未来 6 个月预测: {forecast.ravel().round(0).astype(int)}")
else:
    # 窗口模型 → 需输入特征
    new_data = bundle.train_df.tail(15).copy()[bundle.feature_names]
    pred = inferer.predict(new_data)
    print(f"未来 3 个月预测: {pred[0, :, 0].round(0).astype(int)}")

# 保存 + 加载验证
inferer.save("/tmp/dengue_best_model.zip")
loaded = InferencePipeline.load("/tmp/dengue_best_model.zip")
print(f"模型已保存并验证: {loaded}")
```

---

## 10. 附录

### 可用模型一览

```python
print("Torch 模型:", list_models("torch"))
print("Sklearn 模型:", list_models("sklearn"))
print("TimeSeries 模型:", list_models("ts"))
```

### 依赖

```bash
pip install -e .                      # 基础（Sklearn + TS）
pip install torch                     # 深度学习
pip install -e ".[xgb,lgbm]"         # 额外机器学习
pip install -e ".[all]"              # 全部
```
