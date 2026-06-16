# EpiAI 教程：跨城市泛化能力评估

> 使用气候特征训练模型，评估模型对**已见城市未来疫情**和
> **从未见过的新城市疫情**的预测效果。

---

## 1. 环境准备

```python
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from EpiAI.models import sklearn_models, ts_models
from EpiAI.models import torch_models

from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit,
    Compose, StandardScaler, DateFeatures,
    SlidingWindow, EntityTimeSplit,
)
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import ModelVault

np.random.seed(42)
```

---

## 2. 数据划分

将城市分为两组：
- **已见城市（8 个）**：用于训练和未来预测测试
- **未见城市（2 个）**：完全不在训练集中出现，测试零样本泛化

```python
DF = "../data/China_vector_climate.csv"
df_raw = pd.read_csv(DF)

FEATURES = ["登革热"] + [c for c in df_raw.columns if c.endswith("_mean") and c != "ssrdc_mean"]
TARGET = "登革热"
TIME_COL = "time"

# 按总病例数排序取前 10
prov_ranks = df_raw.groupby("province")["登革热"].sum().sort_values(ascending=False)
top10 = prov_ranks.head(10).index.tolist()

# 分配：8 个已见 + 2 个未见
seen_provinces = top10[:8]      # 广东、云南、福建、广西、浙江、重庆、江西、湖南
unseen_provinces = top10[8:10]  # 四川、海南

df = df_raw[df_raw["province"].isin(top10)].copy()
df = df.rename(columns={"Year/Month": TIME_COL}).reset_index(drop=True)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

print(f"已见城市 ({len(seen_provinces)}): {seen_provinces}")
print(f"未见城市 ({len(unseen_provinces)}): {unseen_provinces}")
print(f"总样本: {len(df)} ({len(df)//len(top10)} 月 × {len(top10)} 市)")

# 分别保存
df[df["province"].isin(seen_provinces)].to_csv("/tmp/seen.csv", index=False)
df[df["province"].isin(unseen_provinces)].to_csv("/tmp/unseen.csv", index=False)
```

---

## 3. 数据管道（已见城市）
## 3. 数据管道（已见城市）

使用 `EntityTimeSplit` 按城市+时间拆分，确保每个城市各自按时间比例划分训练/验证/测试。

```python
df_seen = pd.read_csv("/tmp/seen.csv")
df_seen["time"] = pd.to_datetime(df_seen[TIME_COL])

# 为每个已见城市计算时间分界点
split_map = {}
for city in seen_provinces:
    city_df = df_seen[df_seen["province"] == city].sort_values("time")
    n = len(city_df)
    train_end = city_df.iloc[int(n * 0.7)]["time"]
    val_end = city_df.iloc[int(n * 0.85)]["time"]
    split_map[city] = (str(train_end.date()), str(val_end.date()))

bundle = ForecastPipeline(
    loader=CsvLoader(time_col=TIME_COL, target_cols=TARGET,
                     feature_cols=FEATURES, entity_col="province"),
    split=EntityTimeSplit(split_map=split_map),
    transforms=Compose([
        StandardScaler(columns=FEATURES),
        DateFeatures(time_col=TIME_COL, features=["month"]),
    ]),
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/seen.csv")

print(f"训练窗口: {bundle.train_x.shape}  ({bundle.n_train} 窗)")
print(f"验证窗口: {bundle.val_x.shape}")
print(f"测试窗口: {bundle.test_x.shape}  ({bundle.n_test} 窗, 来自已见城市的未来时段)")
print(f"特征数: {bundle.n_features}")
```

---

## 4. 训练模型

```python
results = {}

# ── Sklearn ──
for name, kwargs in [("RF", {"n_estimators": 200, "max_depth": 10, "random_state": 42}),
                      ("XGB", {"n_estimators": 200, "random_state": 42})]:
    param_key = {"RF": "rf_params", "XGB": "xgb_params"}[name]
    model = get(name)(input_dim=bundle.n_features, lookback=12,
                      horizon=3, target_dim=1, **{param_key: kwargs})
    r = EpiAITrainer(model=model, verbose=False).fit(bundle)
    results[name] = r
    print(f"  ✅ {name:10s}  R²={r.metrics.iloc[0]['R2']:.3f}")

# ── Torch ──
for name in ["MLP", "LSTM"]:
    try:
        model = get(name)(input_dim=bundle.n_features, lookback=12,
                          horizon=3, target_dim=1)
        r = EpiAITrainer(model=model, verbose=False,
                         optimizer_config={"max_epochs": 20}).fit(bundle)
        results[name] = r
        print(f"  ✅ {name:10s}  R²={r.metrics.iloc[0]['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")

# ── TS ──
# 注意：TS 模型不支持多实体，这里仅作为基准
try:
    bundle_ts = ForecastPipeline(
        loader=CsvLoader(time_col=TIME_COL, target_cols=TARGET,
                         feature_cols=TARGET),
        split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
        transforms=None,
        window=SlidingWindow(lookback=12, horizon=3),
    ).run("/tmp/seen.csv")  # 所有城市拼在一起，仅做参考
    for name, kwargs in [("ETS", {"seasonal_periods": 12, "seasonal": "add", "trend": "add"})]:
        r = EpiAITrainer(model=get(name)(**kwargs), verbose=False).fit(bundle_ts)
        results[name] = r
        print(f"  ✅ {name:10s}  R²={r.metrics.iloc[0]['R2']:.3f}")
except Exception as e:
    print(f"  ⚠️ TS 模型跳过: {str(e)[:60]}")
```

---

## 5. 在已见城市的未来测试集上评估

```python
vault = ModelVault.from_results(results, bundle)

print("表 1：已见城市 — 未来时段预测")
print(vault.summary().to_string())
```

---

## 6. 在未见城市上评估（零样本泛化）

将训练好的模型直接用于从未见过的城市。核心操作是**对整个城市的时间序列做滑窗预测，
取每个窗口的第一步行预测作为该时间点的预测值**，然后与真实值对比。

```python
def evaluate_on_city(inferer, city_df, lookback, target_col):
    """
    对单个城市的完整时间序列做滑窗预测。

    逻辑
    ----
    inferer.predict(city_df) 内部：
      apply_features_only 创建 N - lookback + 1 个窗口
      但仅 N - lookback 个窗口的第1步预测落在数据范围内。

    返回
    ----
    y_true : ndarray, (M,)    真实值
    y_pred : ndarray, (M,)    预测值
    """
    pred = inferer.predict(city_df)                # (M', horizon, target_dim)
    n_valid = len(city_df) - lookback              # 有效首步预测数
    y_pred = pred[:n_valid, 0, 0]                  # 仅取有效部分
    y_true = city_df[target_col].values[lookback:][:n_valid]
    return y_true, y_pred

from EpiAI import InferencePipeline
from sklearn.metrics import r2_score, mean_absolute_error

# 加载未见城市数据
df_unseen = pd.read_csv("/tmp/unseen.csv")
df_unseen[TIME_COL] = pd.to_datetime(df_unseen[TIME_COL])

results_unseen = {}
city_results = {}  # {city: {model: {"R2": ..., "MAE": ...}}}

for name in results:
    inferer = InferencePipeline.from_train_result(results[name])
    city_results[name] = {}

    for city in unseen_provinces:
        city_df = df_unseen[df_unseen["province"] == city].reset_index(drop=True)
        if len(city_df) < bundle.lookback + 1:
            continue
        y_true, y_pred = evaluate_on_city(inferer, city_df, bundle.lookback, TARGET)
        city_results[name][city] = {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
        }

    # 汇总（所有城市合并）
    all_true, all_pred = [], []
    for city in city_results[name]:
        y_true, y_pred = evaluate_on_city(inferer,
            df_unseen[df_unseen["province"] == city].reset_index(drop=True),
            bundle.lookback, TARGET)
        all_true.append(y_true)
        all_pred.append(y_pred)
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    results_unseen[name] = {
        "R2": r2_score(all_true, all_pred),
        "MAE": mean_absolute_error(all_true, all_pred),
    }

# 打印按城市细分的评估结果
print("\n未见城市零样本评估（按城市）：")
print(f"{'模型':10s}", end="")
for city in unseen_provinces:
    print(f"  {city:4s} R²", end="")
print()
for name in results:
    print(f"{name:10s}", end="")
    for city in unseen_provinces:
        v = city_results[name].get(city, {})
        print(f"  {v.get('R2', float('nan')):>6.3f}", end="")
    print()

print(f"\n{'模型':10s}  {'汇总 R²':8s}  {'汇总 MAE':8s}")
for name in results:
    print(f"{name:10s}  {results_unseen[name]['R2']:>8.3f}  {results_unseen[name]['MAE']:>8.0f}")
```

---

## 7. 可视化

### 7.1 未见城市预测效果（以第一个未见城市为例）

取其中一个未见城市的全部时段，绘制实际值（黑色粗线）和各模型的预测曲线。

```python
example_city = unseen_provinces[0]
city_df = df_unseen[df_unseen["province"] == example_city].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(16, 5))

# 实际值（黑色粗线，突出轨迹）
ax.plot(city_df[TIME_COL], city_df[TARGET], "-", color="black",
        linewidth=2.5, alpha=0.9, label="实际值", zorder=5)

# 各模型预测
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for idx, (name, r) in enumerate(results.items()):
    inferer = InferencePipeline.from_train_result(r)
    y_true, y_pred = evaluate_on_city(inferer, city_df, bundle.lookback, TARGET)
    r2 = city_results[name].get(example_city, {}).get("R2", float("nan"))
    pred_time = city_df[TIME_COL].values[bundle.lookback:][:len(y_pred)]
    ax.plot(pred_time, y_pred, "s--", label=f"{name} (R²={r2:.3f})",
            color=colors[idx], alpha=0.55, markersize=3, linewidth=1.2)

ax.set_title(f"未见城市「{example_city}」零样本预测效果", fontsize=14, fontweight="bold")
ax.set_ylabel("登革热病例数"); ax.set_xlabel("时间"); ax.grid(alpha=0.15, linestyle=":")
ax.legend(fontsize=9, ncol=4, loc="upper left")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("/tmp/generalization_city.png", dpi=150); plt.show()
```

### 7.2 泛化能力对比柱状图

```python
seen_names = [n for n in results]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(seen_names))
width = 0.35

seen_r2s = [results[n].metrics.iloc[0]["R2"] for n in seen_names]
unseen_r2s = [results_unseen.get(n, {}).get("R2", 0) for n in seen_names]

bars1 = ax.bar(x - width/2, seen_r2s, width, label="已见城市（未来时段）",
               color="steelblue", alpha=0.85)
bars2 = ax.bar(x + width/2, unseen_r2s, width, label="未见城市（零样本）",
               color="coral", alpha=0.85)

ax.set_ylabel("R²"); ax.set_title("跨城市泛化能力对比", fontsize=14, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(seen_names)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(min(min(seen_r2s), min(unseen_r2s), 0) - 0.2, 1.05)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.02,
            f"{bar.get_height():.2f}", ha="center", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.02,
            f"{bar.get_height():.2f}", ha="center", fontsize=8)

plt.tight_layout(); plt.savefig("/tmp/generalization_bar.png", dpi=150); plt.show()
```

---

## 8. 结论

```python
print("=== 核心发现 ===")
print()
print(f"已见城市测试集 (n={bundle.n_test} 窗口):")
for name in results:
    r2 = results[name].metrics.iloc[0]["R2"]
    print(f"  {name}: R²={r2:.3f}")

print(f"\n未见城市零样本 (n={len(unseen_provinces)} 城市):")
for name in results:
    if name in results_unseen:
        r2 = results_unseen[name]["R2"]
        mae = results_unseen[name]["MAE"]
        print(f"  {name}: R²={r2:.3f}  MAE={mae:.0f}")
```

> 模型在已见城市的未来时段表现较好，原因是可以利用训练数据中学到的
> 季节性和气候-发病关系。对于未见城市，RF 等非参数模型通常比深度学习
> 模型泛化更稳定，因为决策树对特征的单调关系假设更少。
