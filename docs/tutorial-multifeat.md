# EpiAI 教程：多特征登革热预测

> 使用气候特征（温度、降水、风速等）预测广东省登革热月发病数。
> 展示多特征输入下数据管道、模型训练与部署全流程。

---

## 1. 环境准备

```python
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from EpiAI.models import sklearn_models, ts_models
from EpiAI.models import torch_models          # 需要 PyTorch

from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, Compose,
    StandardScaler, DateFeatures, FeatureLag,
    SlidingWindow,
)
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import ModelVault
```

---

## 2. 数据探索

```python
DF = "../data/China_vector_climate.csv"
df_raw = pd.read_csv(DF)

# 筛选广东省
df = df_raw[df_raw["province"] == "广东"].copy()
df = df.rename(columns={"Year/Month": "time"}).reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"])

print(f"广东登革热: {len(df)} 个月 ({df['time'].min().date()} ~ {df['time'].max().date()})")
print(f"病例数: min={df['登革热'].min()}, max={df['登革热'].max()}, "
      f"mean={df['登革热'].mean():.0f}, nonzero={ (df['登革热']>0).sum()}/{len(df)}")

# 查看气候特征与登革热的关系
feature_cols = ["t2m_mean", "tp_mean", "tcc_mean", "ws10_mean", "skt_mean"]
fig, axes = plt.subplots(len(feature_cols), 1, figsize=(14, 10), sharex=True)
axes[0].plot(df["time"], df["登革热"], color="black", linewidth=1.5, label="登革热")
axes[0].set_ylabel("病例数"); axes[0].legend(); axes[0].grid(alpha=0.3)

for ax, col in zip(axes[1:], feature_cols):
    ax.plot(df["time"], df[col], color="steelblue", alpha=0.7)
    ax.set_ylabel(col); ax.grid(alpha=0.3)

plt.xlabel("时间"); plt.tight_layout(); plt.show()
```

---

## 3. 数据管道

使用气候特征（mean 值）+ 登革热历史作为输入特征。

```python
# 选择特征：登革热历史 + 气候均值
FEATURES = ["登革热"] + [c for c in df.columns if c.endswith("_mean") and c != "ssrdc_mean"]
TARGET = "登革热"
TIME_COL = "time"

print(f"特征数: {len(FEATURES)}")
print(f"特征: {FEATURES[:5]}...{FEATURES[-3:]}")

# 保存广东数据
df_gd = df[["time"] + FEATURES].copy()
df_gd.to_csv("/tmp/guangdong.csv", index=False)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col=TIME_COL, target_cols=TARGET,
                     feature_cols=FEATURES),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=Compose([
        StandardScaler(columns=FEATURES),              # 所有特征标准化
        DateFeatures(time_col=TIME_COL, features=["month"]),  # 月份特征
        FeatureLag(columns=["登革热"], lags=[1, 2, 3, 12]),   # 登革热滞后特征
    ]),
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/guangdong.csv")

print(f"训练: {bundle.train_x.shape}  验证: {bundle.val_x.shape}  测试: {bundle.test_x.shape}")
print(f"实际特征数 (含变换后): {bundle.n_features}")
```

---

## 4. 训练模型

### 4.1 深度学习（Torch）

```python
results = {}

for name in ["MLP", "LSTM"]:
    try:
        model = get(name)(input_dim=bundle.n_features, lookback=12,
                          horizon=3, target_dim=1)
        r = EpiAITrainer(model=model, verbose=False,
                         optimizer_config={"max_epochs": 20}).fit(bundle)
        results[name] = r
        m = r.metrics.iloc[0]
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")
```

### 4.2 机器学习（Sklearn）

```python
for name, kwargs in [
    ("RF",  {"n_estimators": 200, "max_depth": 10, "random_state": 42}),
    ("XGB", {"n_estimators": 200, "random_state": 42}),
    ("SVR", {"kernel": "rbf", "C": 1.0}),
]:
    try:
        param_key = {"RF": "rf_params", "XGB": "xgb_params", "SVR": "svm_params"}[name]
        model = get(name)(input_dim=bundle.n_features, lookback=12,
                          horizon=3, target_dim=1, **{param_key: kwargs})
        r = EpiAITrainer(model=model, verbose=False).fit(bundle)
        results[name] = r
        m = r.metrics.iloc[0]
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")
```

### 4.3 时间序列（TimeSeries）

TS 模型不使用气候特征（纯单变量）。

```python
# 为 TS 模型构建无特征的管道
bundle_ts = ForecastPipeline(
    loader=CsvLoader(time_col=TIME_COL, target_cols=TARGET,
                     feature_cols=TARGET),       # 仅用登革热
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,                              # 无变换
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/guangdong.csv")

for name, kwargs in [("ETS", {"seasonal_periods": 12, "seasonal": "add", "trend": "add"}),
                      ("ARIMA", {"seasonal": True, "m": 12})]:
    try:
        model = get(name)(**kwargs)
        r = EpiAITrainer(model=model, verbose=False).fit(bundle_ts)
        results[name] = r
        m = r.metrics.iloc[0]
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")
```

---

## 5. 模型对比

```python
vault = ModelVault.from_results(results, bundle)
vault.save("/tmp/guangdong_vault/")

print("\n模型对比总表（含气候特征的模型 vs 纯时间序列）：")
print(vault.summary().to_string())

best_name = vault.best("R2")
print(f"\n最佳模型: {best_name}")
```

---

## 6. 预测可视化

```python
all_time = bundle.train_df[TIME_COL].tolist() + bundle.test_df[TIME_COL].tolist()
y_all = np.concatenate([bundle.get_y_series("train").ravel(),
                        bundle.get_y_series("test").ravel()])

plt.figure(figsize=(14, 7))

# 训练集
plt.plot(all_time[:len(bundle.train_df)], y_all[:len(bundle.train_df)],
         "-", label="训练集", color="#bdc3c7", alpha=0.5)

# 测试集实际
test_start = len(bundle.train_df)
test_time = all_time[test_start:]
test_actual = bundle.get_y_series("test").ravel()
plt.plot(test_time, test_actual, "o-", label="实际值", color="black", linewidth=2)

# 模型预测
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for (name, r), color in zip(results.items(), colors):
    preds = r.predictions[:, 0, 0]
    n = len(preds)
    m = r.metrics.iloc[0]
    plt.plot(test_time[:n], preds, "s--",
             label=f"{name}  (R²={m['R2']:.3f})", color=color, alpha=0.6, markersize=4)

plt.legend(fontsize=9, ncol=3)
plt.title("广东省登革热 — 多特征预测对比", fontsize=14)
plt.ylabel("病例数"); plt.xlabel("时间"); plt.grid(alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("/tmp/guangdong_all_models.png", dpi=150)
plt.show()
```

---

## 7. 特征重要性分析（以 RF 为例）

```python
rf_inferer = vault["RF"]
rf_model = rf_inferer.model.model_

# 获取变换后的特征名
print(f"RF 特征重要性 (Top 15):")
importances = rf_model.feature_importances_
if hasattr(rf_inferer.transforms, "get_feature_names"):
    feature_names = rf_inferer.transforms.get_feature_names()
else:
    feature_names = [f"f{i}" for i in range(len(importances))]

idx = np.argsort(importances)[::-1][:15]
for i in idx:
    print(f"  {feature_names[i]:30s}  {importances[i]:.4f}")
```

> 注意：做了 FeatureLag 和 DateFeatures 后特征列数会变多，
> 可以通过 `bundle.n_features` 查看变换后的实际特征数。

---

## 8. 与无气候特征的基准对比

```python
# 上面 bundle_ts 的 results 中包含 ARIMA/ETS 的结果
# 这些是纯单变量模型，可以对比看气候特征是否提升效果

print("\n气候特征模型 vs 纯时序模型：")
for name in results:
    r = results[name]
    m = r.metrics.iloc[0]
    has_features = r.predictions.shape[-1] == bundle.n_features or name not in ["ETS", "ARIMA"]
    tag = "🌤 含气候" if name not in ["ETS", "ARIMA"] else "📈 纯时序"
    print(f"  {tag} {name:8s}  R²={m['R2']:.3f}  MAE={m['MAE']:.0f}")
```

---

## 9. 部署

```python
from EpiAI import DeploymentRuntime

runtime = DeploymentRuntime(vault, time_col=TIME_COL, time_unit="MS")
runtime.data_table = df_gd.copy()
runtime._train_end_time = df_gd[TIME_COL].iloc[-1]

# 模拟最后一次预测
last_row = df_gd.iloc[-1:].copy()
last_row[TIME_COL] = (pd.to_datetime(last_row[TIME_COL].iloc[0])
                      + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
# 填入最近的气候数据（这里用最后一条的均值近似）
for col in FEATURES:
    if col != "登革热":
        last_row[col] = df_gd[col].iloc[-12:].mean()

result = runtime.feed(last_row)
print("\n部署预测：")
for name, r in result.items():
    if "error" not in r:
        print(f"  {name}: {[t.strftime('%Y-%m') for t in r['time']]} → "
              f"{r['pred'].round(0).astype(int)}")
```

---

## 10. 附录

```python
print("可用模型:", list_models())
```
