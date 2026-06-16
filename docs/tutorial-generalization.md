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
    ForecastPipeline, CsvLoader, TimeSplit, EntitySplit,
    Compose, StandardScaler, DateFeatures, FeatureLag,
    SlidingWindow,
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

使用 `EntitySplit` 按城市拆分，确保训练/验证/测试集不混合同一城市的时序。

```python
# 对已见城市：70% 时间用于训练，后 30% 用于测试
bundle = ForecastPipeline(
    loader=CsvLoader(time_col=TIME_COL, target_cols=TARGET,
                     feature_cols=FEATURES, entity_col="province"),
    split=EntityTimeSplit(        # 按实体+时间拆分
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        entity_col="province",
    ),
    transforms=Compose([
        StandardScaler(columns=FEATURES),
        DateFeatures(time_col=TIME_COL, features=["month"]),
        FeatureLag(columns=["登革热"], lags=[1, 2, 3, 6, 12]),
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

将训练好的模型直接用于从未见过的城市，不做任何微调。

```python
# 对未见城市的全部数据做预测
df_unseen = pd.read_csv("/tmp/unseen.csv")
df_unseen[TIME_COL] = pd.to_datetime(df_unseen[TIME_COL])

# 获取每个已见城市最后一个训练时段的模型，用于预测未见城市
# 实际会使用 bundle 中的测试数据进行评估（模型的 test_predictions）
# 但对于未见城市，我们需要用 InferencePipeline 单独预测

from EpiAI import InferencePipeline

results_unseen = {}

for name in results:
    inferer = InferencePipeline.from_train_result(results[name])
    preds_city = []
    actuals_city = []
    city_names = []

    for city in unseen_provinces:
        city_df = df_unseen[df_unseen["province"] == city].reset_index(drop=True)
        if len(city_df) < bundle.lookback:
            continue

        # 用全部数据预测（模型从未见过这个城市）
        pred = inferer.predict(city_df)

        y_true_city = bundle.get_y_series("test")  # not applicable here
        y_actual = city_df[TARGET].values[bundle.lookback:][:pred.shape[0]]

        preds_city.append(pred[:, 0, 0])
        actuals_city.append(y_actual)
        city_names.extend([city] * pred.shape[0])

    if preds_city:
        all_preds = np.concatenate(preds_city)
        all_actual = np.concatenate(actuals_city)

        from sklearn.metrics import r2_score, mean_absolute_error

        results_unseen[name] = {
            "R2": r2_score(all_actual, all_preds),
            "MAE": mean_absolute_error(all_actual, all_preds),
        }
        print(f"  {name:10s}  R²={results_unseen[name]['R2']:.3f}  "
              f"MAE={results_unseen[name]['MAE']:.0f}")
```

---

## 7. 对比：已见 vs 未见城市

```python
print("\n表 2：泛化能力对比")
print(f"{'模型':10s}  {'已见城市 R²':12s}  {'未见城市 R²':12s}  {'差距':8s}")
print("-" * 45)
for name in results:
    seen_r2 = results[name].metrics.iloc[0]["R2"]
    unseen_r2 = results_unseen.get(name, {}).get("R2", float("nan"))
    gap = seen_r2 - unseen_r2 if not np.isnan(unseen_r2) else float("nan")
    print(f"{name:10s}  {seen_r2:>10.3f}      {unseen_r2:>8.3f}      {gap:>+.3f}" if not np.isnan(gap)
          else f"{name:10s}  {seen_r2:>10.3f}      {'N/A':>8}      {'N/A':>8}")
```

---

## 8. 可视化

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: seen city future prediction
all_time = bundle.train_df[TIME_COL].tolist() + bundle.test_df[TIME_COL].tolist()
y_all = np.concatenate([bundle.get_y_series("train").ravel(),
                        bundle.get_y_series("test").ravel()])

# 取其中一个已见城市的测试时段（通过实体列筛选）
test_mask = bundle.test_df["province"] == seen_provinces[0]
test_time = bundle.test_df[TIME_COL][test_mask].values
test_actual = bundle.test_df[TARGET][test_mask].values

# Right: unseen city prediction
unseen_city = unseen_provinces[0]
city_df = df_unseen[df_unseen["province"] == unseen_city].reset_index(drop=True)

for ax, title in zip(axes, [f"已见城市 — {seen_provinces[0]} 未来预测",
                              f"未见城市 — {unseen_city} 零样本预测"]):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("时间"); ax.set_ylabel("登革热病例数"); ax.grid(alpha=0.3)

# 已见城市
axes[0].plot(test_time, test_actual, "o-", color="black", linewidth=2, label="实际")
for (name, r), color in zip(results.items(), plt.cm.tab10(np.linspace(0, 1, len(results)))):
    preds = r.predictions[:, 0, 0]
    m = r.metrics.iloc[0]["R2"]
    # 按实体匹配预测
    axes[0].plot(test_time[:len(preds)], preds, "s--",
                 label=f"{name} (R²={m:.3f})", color=color, alpha=0.6)
axes[0].legend(fontsize=8, ncol=2); axes[0].tick_params(axis="x", rotation=45)

# 未见城市
y_actual_unseen = city_df[TARGET].values
axes[1].plot(city_df[TIME_COL], y_actual_unseen, "-", color="black", alpha=0.4, label="实际")

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for idx, (name, r) in enumerate(results.items()):
    inferer = InferencePipeline.from_train_result(r)
    pred = inferer.predict(city_df)
    pred_vals = pred[:, 0, 0]
    pred_time = city_df[TIME_COL].values[bundle.lookback:][:len(pred_vals)]

    r2 = results_unseen.get(name, {}).get("R2", float("nan"))
    axes[1].plot(pred_time, pred_vals, "s--",
                 label=f"{name} (R²={r2:.3f})", color=colors[idx], alpha=0.6)

axes[1].legend(fontsize=8, ncol=2); axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("/tmp/generalization.png", dpi=150); plt.show()
```

---

## 9. 结论

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
