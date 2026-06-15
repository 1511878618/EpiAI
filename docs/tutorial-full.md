# EpiAI 完整流程教程

> 从数据加载到生产部署的端到端示例。留出最后 12 个月的真实数据
> 作为部署模拟数据，直观展示「训练 → 部署 → 逐月 feed → 对比验证」。

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
    StandardScaler, Log1pTransform, DateFeatures,
    FeatureLag, SlidingWindow,
)
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import InferencePipeline, ModelVault, DeploymentRuntime

print(f"已注册模型: {len(list_models())}")
```

---

## 2. 数据加载与拆留

留出最后 12 个月作为部署验证数据，前面的用于训练。

```python
CSV = "../data/Infective_disease_china-V3.csv"
df_raw = pd.read_csv(CSV)

# 筛选登革热
df = df_raw[df_raw["Diseases"] == "登革热 Dengue fever"].copy()
df = df.rename(columns={"Year/Month": "time", "Case number": "cases"})
df = df[["time", "cases"]].reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"])
df["cases"] = df["cases"].astype(float)

print(f"登革热: {len(df)} 个月 ({df['time'].min().date()} ~ {df['time'].max().date()})")
print(f"病例数: 最小={df['cases'].min():.0f}, 最大={df['cases'].max():.0f}, "
      f"均值={df['cases'].mean():.0f}")

# ── 留出最后 12 个月作为部署模拟数据 ──
N_HELD_OUT = 12
df_train = df.iloc[:-N_HELD_OUT].copy()     # 用于训练
df_deploy = df.iloc[-N_HELD_OUT:].copy()    # 用于部署验证

print(f"\n训练数据: {len(df_train)} 个月 ({df_train['time'].min().date()} ~ {df_train['time'].max().date()})")
print(f"部署数据: {len(df_deploy)} 个月 ({df_deploy['time'].min().date()} ~ {df_deploy['time'].max().date()})  ← 留出的验证集")

plt.figure(figsize=(14, 4))
plt.plot(df["time"], df["cases"], color="#2c3e50", label="全部数据")
plt.axvline(x=df_deploy["time"].iloc[0], color="red", linestyle="--", alpha=0.6, label="部署验证开始")
plt.title("全国登革热月发病数 (2010–2026)", fontsize=13)
plt.ylabel("病例数"); plt.legend(); plt.grid(alpha=0.3); plt.show()
```

---

## 3. 数据管道

```python
df_train.to_csv("/tmp/dengue_train.csv", index=False)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols="cases"),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/dengue_train.csv")

print(f"训练窗口: {bundle.train_x.shape}  验证: {bundle.val_x.shape}  测试: {bundle.test_x.shape}")
print(f"lookback={bundle.lookback}, horizon={bundle.horizon}, features={bundle.n_features}")
```

---

## 4. 模型训练

### 4.1 深度学习（Torch）

```python
results = {}

for name in ["MLP", "LSTM", "CNN"]:
    try:
        model = get(name)(input_dim=bundle.n_features, lookback=12,
                          horizon=3, target_dim=1)
        result = EpiAITrainer(model=model, verbose=False,
                              optimizer_config={"max_epochs": 10}).fit(bundle)
        results[name] = result
        m = result.metrics.iloc[0]
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:10s}  {str(e)[:50]}")
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
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        results[name] = result
        m = result.metrics.iloc[0]
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:10s}  {str(e)[:50]}")
```

### 4.3 时间序列（TimeSeries）

```python
for name, kwargs in [
    ("ETS",   {"seasonal_periods": 12, "seasonal": "add", "trend": "add"}),
    ("ARIMA", {"seasonal": True, "m": 12}),
]:
    try:
        model = get(name)(**kwargs)
        result = EpiAITrainer(model=model, verbose=False).fit(bundle)
        results[name] = result
        m = result.metrics.iloc[0]
        print(f"  ✅ {name:10s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}")
    except Exception as e:
        print(f"  ❌ {name:10s}  {str(e)[:50]}")
```

---

## 5. ModelVault：模型入库与对比

```python
vault = ModelVault.from_results(results, bundle)
vault.save("/tmp/dengue_vault/")

print("\n模型对比总表（按 R² 排序）：")
print(vault.summary().to_string())

best_name = vault.best("R2")
print(f"\n最佳模型: {best_name}")
```

---

## 6. 可视化：历史与预测

```python
all_time = bundle.train_df["time"].tolist() + bundle.test_df["time"].tolist()
y_all = np.concatenate([bundle.get_y_series("train").ravel(),
                        bundle.get_y_series("test").ravel()])

plt.figure(figsize=(14, 7))
plt.plot(all_time[:len(bundle.train_df)], y_all[:len(bundle.train_df)],
         "-", label="训练集", color="#bdc3c7", alpha=0.5)
test_start = len(bundle.train_df)
test_time = all_time[test_start:]
test_actual = bundle.get_y_series("test").ravel()
plt.plot(test_time, test_actual, "o-", label="实际值", color="black", linewidth=2)

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for (name, r), color in zip(results.items(), colors):
    preds = r.predictions[:, 0, 0]
    n = len(preds)
    m = r.metrics.iloc[0]
    plt.plot(test_time[:n], preds, "s--", label=f"{name}  (R²={m['R2']:.3f})",
             color=color, alpha=0.6, markersize=4)

plt.legend(fontsize=9, ncol=3)
plt.title("登革热月发病数 — 多模型预测对比", fontsize=14)
plt.ylabel("病例数"); plt.xlabel("时间"); plt.grid(alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("/tmp/dengue_all_models.png", dpi=150)
plt.show()
```

---

## 7. 生产部署模拟

用留出的 12 个月真实数据模拟生产环境逐月 feed。
初始化时，将训练数据全部加载到 `data_table` 中，这样窗口模型和时序模型都有完整的历史可查。

```python
runtime = DeploymentRuntime(
    vault=vault,
    time_col="time",
    time_unit="MS",
    strict=True,
)
# 将全部训练数据载入 data_table（含 transforms=None 时的原始值）
runtime.data_table = df_train.copy()
runtime._train_end_time = df_train["time"].iloc[-1]
print(f"训练数据: {len(runtime.data_table)} 行")
print(f"训练结束: {runtime._train_end_time.date()}")
print(f"部署开始: {df_deploy['time'].iloc[0].date()}")
print(f"窗口模型需要: {bundle.lookback} 行历史 ✅ data_table 已满足")

# ── 逐月 feed 留出的真实数据 ──
history = []

for i in range(len(df_deploy)):
    row = df_deploy.iloc[i]
    new_data = pd.DataFrame({
        "time": [row["time"].strftime("%Y-%m-%d")],
        "cases": [row["cases"]],
    })

    result = runtime.feed(new_data)

    month_label = row["time"].strftime("%Y-%m")
    actual = int(row["cases"])

    record = {"month": month_label, "actual": actual}
    for name, r in result.items():
        if "error" not in r:
            record[name] = r["pred"][0]
    history.append(record)

    print(f"  {month_label}: 实报={actual:5d}", end="")
    for name, r in result.items():
        if "error" not in r:
            print(f"  {name}预测={r['pred'][0]:6.0f}", end="")
    print()

print(f"\n模拟完成: {len(df_deploy)} 个月, data_table={len(runtime.data_table)} 行")
```

### 7.1 部署预测 vs 实际值对比图

```python
df_hist = pd.DataFrame(history)

plt.figure(figsize=(14, 5))
plt.plot(range(len(df_hist)), df_hist["actual"], "o-", color="black",
         linewidth=2, label="实际值", markersize=6)

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for (name, r), color in zip(results.items(), colors):
    if name in df_hist.columns:
        plt.plot(range(len(df_hist)), df_hist[name], "s--", label=name,
                 color=color, alpha=0.6, markersize=4)

plt.xticks(range(len(df_hist)), df_hist["month"], rotation=45)
plt.ylabel("病例数"); plt.xlabel("月份")
plt.title("部署模拟：逐月预测 vs 实际值", fontsize=14, fontweight="bold")
plt.legend(fontsize=10); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/dengue_deploy_sim.png", dpi=150)
plt.show()
```

### 7.2 保存运行时

```python
runtime.save("/tmp/dengue_runtime/")
print("运行时已保存到 /tmp/dengue_runtime/")

# 保存 vault
vault.save("/tmp/dengue_vault/")
print("模型 vault 已保存到 /tmp/dengue_vault/")
```

---

## 8. 未来预测

```python
best_name = vault.best("R2")
inferer = vault.get(best_name)

last_time = pd.to_datetime(runtime.data_table["time"].iloc[-1])
future_dates = pd.date_range(start=last_time + pd.DateOffset(months=1),
                              periods=3, freq="MS")

if inferer.paradigm == "ts":
    fc = inferer.forecast(3)
    future_pred = fc[:, 0, 0]
else:
    pred = inferer.predict(
        runtime.data_table.tail(bundle.lookback)[bundle.feature_names]
    )
    future_pred = pred[0, :, 0]

# 历史数据（训练 + 部署验证）
all_history = df["time"].tolist()
all_cases = df["cases"].values

plt.figure(figsize=(14, 5))
plt.plot(all_history, all_cases, "-", label="历史实际值", color="#2c3e50", linewidth=1.5)
plt.plot(future_dates, future_pred, "o--", color="#e74c3c",
         linewidth=2, markersize=6, label=f"预测 (3个月)")

# 部署验证区域标注
deploy_start = df_deploy["time"].iloc[0]
plt.axvspan(deploy_start, all_history[-1], alpha=0.08, color="blue", label="部署验证期")
plt.axvline(x=last_time, color="gray", linestyle=":", alpha=0.5)
plt.text(last_time, plt.ylim()[1] * 0.95, "← 历史 | 预测 →",
         ha="center", fontsize=10, color="gray")

plt.legend(fontsize=11)
plt.title(f"{best_name} — 登革热未来 3 个月预测", fontsize=14, fontweight="bold")
plt.ylabel("病例数"); plt.xlabel("时间"); plt.grid(alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("/tmp/dengue_forecast.png", dpi=150)
plt.show()
```

---

## 9. 恢复运行时继续预测

```python
runtime2 = DeploymentRuntime.load("/tmp/dengue_runtime/")
print(f"加载: {runtime2}")

# 继续 feed 新数据
new_row = pd.DataFrame({
    "time": [(last_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d")],
    "cases": [int(np.random.normal(1000, 300))],
})
result = runtime2.feed(new_row)
for name, r in result.items():
    if "error" not in r:
        print(f"  继续 feed — {name}: 预测未来3月 = {r['pred'].round(0).astype(int)}")
```

---

## 10. 附录

```python
print("可用模型:")
print(f"  Torch:   {list_models('torch')}")
print(f"  Sklearn: {list_models('sklearn')}")
print(f"  TimeSeries: {list_models('ts')}")
```

依赖安装：

```bash
pip install -e .                                # 基础
pip install torch                               # 深度学习
pip install -e ".[xgb,lgbm]"                   # 额外机器学习
pip install -e ".[all]"                        # 全部
```
