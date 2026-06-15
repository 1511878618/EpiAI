# EpiAI 使用教程：全国登革热月发病数预测

> 纯自回归（用历史病例预测未来病例），对比深度学习、机器学习、时间序列三类模型。

---

## 1. 环境准备

```python
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from EpiAI.models import sklearn_models, ts_models
try:
    from EpiAI.models import torch_models  # 可选，需要 torch
except ImportError:
    torch_models = None

from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, SlidingWindow,
)
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import InferencePipeline
```

---

## 2. 数据加载

```python
df_raw = pd.read_csv("../data/Infective_disease_china-V3.csv")
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

纯自回归：用历史 `cases` 预测未来 `cases`。

```python
df.to_csv("/tmp/dengue.csv", index=False)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols="cases"),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/dengue.csv")

print(f"训练: {bundle.train_x.shape}  ({bundle.n_train} 个窗口)")
print(f"验证: {bundle.val_x.shape}     ({bundle.n_val} 个窗口)")
print(f"测试: {bundle.test_x.shape}     ({bundle.n_test} 个窗口)")
```

---

## 4. 模型训练与对比

### 4.1 窗口模型（Torch + Sklearn 族）

```python
results = []

# 深度学习（需 torch）
window_names = ["MLP", "LSTM", "CNN"] if torch_models else []
# 机器学习
window_names += ["RF", "XGB", "LGBM", "SVR", "GLM"]

for name in window_names:
    try:
        model_cls = get(name)
    except KeyError:
        continue

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
        model = model_cls(
            input_dim=bundle.n_features, lookback=12,
            horizon=3, target_dim=1, **extra,
        )
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        m = result.metrics.iloc[0]
        results.append({"model": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                        "R2": m["R2"], "PearsonR": m["PearsonR"]})
        print(f"  ✅ {name:8s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}  r={m['PearsonR']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:8s}  {str(e)[:60]}")
```

### 4.2 时序模型（TimeSeries 族）

```python
ts_names = ["ETS", "ARIMA"]

for name in ts_names:
    try:
        model_cls = get(name)
    except KeyError:
        continue

    try:
        extra = {"seasonal_periods": 12, "seasonal": "add", "trend": "add"} if name == "ETS" \
                else {"seasonal": True, "m": 12}
        model = model_cls(**extra)
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        m = result.metrics.iloc[0]
        results.append({"model": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                        "R2": m["R2"], "PearsonR": m["PearsonR"]})
        print(f"  ✅ {name:8s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}  r={m['PearsonR']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:8s}  {str(e)[:60]}")
```

### 4.3 结果汇总

```python
comp = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n模型对比（按 R² 排序）：")
print(comp.to_string(index=False))
```

---

## 5. 预测结果可视化

```python
best_name = comp.iloc[0]["model"]
print(f"最佳模型: {best_name}")

# 重训最佳模型
def _build_model(name, bundle):
    extra = {}
    if name == "RF":   extra["rf_params"] = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    elif name == "XGB": extra["xgb_params"] = {"n_estimators": 200, "random_state": 42}
    elif name == "ETS": return get(name)(seasonal_periods=12, seasonal="add", trend="add")
    elif name == "ARIMA": return get(name)(seasonal=True, m=12)
    return get(name)(input_dim=bundle.n_features, lookback=12, horizon=3, target_dim=1, **extra)

best_model = _build_model(best_name, bundle)
best_result = EpiAITrainer(model=best_model, verbose=False).fit(bundle)
best_result._bundle = bundle

# 提取预测结果
y_test = bundle.get_y_series("test")
n = best_result.predictions.shape[0]
if best_result.predictions.shape[2] == 1:
    y_pred = best_result.predictions[:, -1, 0]   # (N,)
else:
    y_pred = best_result.predictions[:, -1, 0]   # (N,)
y_true = y_test[:n, 0]

# 绘图
plt.figure(figsize=(12, 5))
plt.plot(y_true, "o-", label="实际值", color="#2c3e50", alpha=0.8)
plt.plot(y_pred, "s--", label=f"预测值 ({best_name})", color="#e74c3c", alpha=0.7)
plt.legend(fontsize=12)
plt.xlabel("测试样本序号", fontsize=11)
plt.ylabel("登革热病例数", fontsize=11)
plt.title(f"{best_name} — 登革热月发病数预测", fontsize=13, fontweight="bold")
plt.grid(alpha=0.3)

# 标注指标
m = best_result.metrics.iloc[0]
txt = f"MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  R²={m['R2']:.3f}  r={m['PearsonR']:.3f}"
plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig("/tmp/dengue_prediction.png", dpi=150)
plt.show()
print("图表已保存: /tmp/dengue_prediction.png")
```

---

## 6. 模型部署与推理

不同模型族的推理方式不同，根据最佳模型的 paradigm 自动选择：

```python
inferer = InferencePipeline.from_train_result(best_result)
print(f"部署: {inferer}")

paradigm = best_result.model.paradigm()

if paradigm == "ts":
    # 时序模型 → 纯未来预测（不需要输入特征）
    forecast = inferer.forecast(steps=6)
    print(f"\n未来 6 个月预测: {forecast.ravel().round(0).astype(int)}")

else:
    # 窗口模型（torch/sklearn）→ 需要输入特征
    new_data = bundle.train_df.tail(15).copy()[bundle.feature_names]
    pred = inferer.predict(new_data)
    print(f"\n预测未来 3 个月: {pred[0, :, 0].round(0).astype(int)}")

# 保存
inferer.save("/tmp/dengue_best_model.zip")
print("模型已保存: /tmp/dengue_best_model.zip")

# 加载验证
loaded = InferencePipeline.load("/tmp/dengue_best_model.zip")
print(f"加载验证: {loaded}")
```

---

## 7. 全部模型对比图

```python
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

# 为每个模型生成预测并绘图
for idx, row in comp.iterrows():
    if idx >= 6:
        break
    name = row["model"]
    m = _build_model(name, bundle)
    r = EpiAITrainer(model=m, verbose=False).fit(bundle)
    n = r.predictions.shape[0]
    yp = r.predictions[:, -1, 0]
    yt = bundle.get_y_series("test")[:n, 0]

    ax = axes[idx]
    ax.plot(yt, "o-", label="实际", color="#2c3e50", alpha=0.6, ms=3)
    ax.plot(yp, "s--", label="预测", color="#e74c3c", alpha=0.6, ms=3)
    ax.set_title(f"{name}  (R²={row['R2']:.3f})", fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/dengue_all_models.png", dpi=150)
plt.show()
```

---

## 8. 附录

```python
print("Torch 模型:", list_models("torch"))
print("Sklearn 模型:", list_models("sklearn"))
print("TimeSeries 模型:", list_models("ts"))
```

依赖安装：

```bash
pip install -e .                      # 基础
pip install -e ".[xgb,lgbm]"         # 机器学习
pip install -e ".[all]"              # 全部（含 torch）
```
