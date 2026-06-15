# EpiAI 完整流程教程

> 从数据加载到生产部署的端到端示例 —— 全国登革热月发病数预测。

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

## 2. 数据加载

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

plt.figure(figsize=(14, 4))
plt.plot(df["time"], df["cases"], color="#2c3e50")
plt.title("全国登革热月发病数 (2010–2026)", fontsize=13)
plt.ylabel("病例数"); plt.grid(alpha=0.3); plt.show()
```

---

## 3. 数据管道

纯自回归（用历史病例预测未来病例），不做特征工程，让 baseline 清晰。

```python
df.to_csv("/tmp/dengue.csv", index=False)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols="cases"),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,                                # 无变换
    window=SlidingWindow(lookback=12, horizon=3),   # 用过去一年预测未来一季度
).run("/tmp/dengue.csv")

print(f"训练窗口: {bundle.train_x.shape}  验证: {bundle.val_x.shape}  测试: {bundle.test_x.shape}")
print(f"lookback={bundle.lookback}, horizon={bundle.horizon}, features={bundle.n_features}")
```

如果用标准化的数据管道（可选的）：

```python
bundle2 = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols="cases"),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=Compose([
        Log1pTransform(columns=["cases"]),             # 对数变换（偏态分布）
        StandardScaler(columns=["cases"]),              # 标准化
        DateFeatures(time_col="time", features=["month", "season"]),  # 时间特征
        FeatureLag(columns=["cases"], lags=[1, 2, 3, 6, 12]),        # 滞后特征
    ]),
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/dengue.csv")
print(f"特征管道: {bundle2.n_features} 个特征")
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
# 时间轴
all_time = bundle.train_df["time"].tolist() + bundle.test_df["time"].tolist()
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

# 各模型预测
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

## 7. 生产部署

```python
# 初始化运行时
runtime = DeploymentRuntime(
    vault=vault,
    time_col="time",
    time_unit="MS",
    strict=True,
)

# 设置训练结束时间（首次 feed 会检查连续性）
runtime._train_end_time = pd.to_datetime("2026-03-01")

# 模拟：每月新数据到达，逐月 feed
np.random.seed(42)
for i in range(14):  # 14 个月（含 12 个 lookback + 2 个预测验证）
    month = pd.Timestamp("2026-04-01") + pd.DateOffset(months=i)
    new_cases = int(np.random.normal(1000, 300))  # 模拟真实上报
    new_row = pd.DataFrame({"time": [month.strftime("%Y-%m-%d")],
                            "cases": [new_cases]})

    result = runtime.feed(new_row)
    print(f"{month.strftime('%Y-%m')}: 实报={new_cases:5d}", end="")
    for name, r in result.items():
        if "error" not in r:
            print(f"  {name}预测={r['pred'][0]:.0f}", end="")
    print()

# 最后一次 feed 后查看 data_table
print(f"\ndata_table: {len(runtime.data_table)} 行")
print(f"时间范围: {runtime.data_table['time'].iloc[0]} ~ {runtime.data_table['time'].iloc[-1]}")

# 保存运行时
runtime.save("/tmp/dengue_runtime/")
print(f"运行时已保存到 /tmp/dengue_runtime/")

# 显式更新 ETS 状态（确认数据质量后）
runtime.update_model("ETS", np.array([1200, 800]))
print("\nETS 状态已更新")
```

---

## 8. 未来预测图

```python
# 选择最佳模型绘制未来预测
best_name = vault.best("R2")
inferer = vault.get(best_name)

last_time = pd.to_datetime(all_time[-1])
future_dates = pd.date_range(start=last_time + pd.DateOffset(months=1),
                              periods=3, freq="MS")

if inferer.paradigm == "ts":
    fc = inferer.forecast(3)
    future_pred = fc[:, 0, 0]
else:
    pred = inferer.predict(bundle.train_df.tail(15).copy()[bundle.feature_names])
    future_pred = pred[0, :, 0]

plt.figure(figsize=(14, 5))
plt.plot(all_time, y_all, "-", label="历史实际值", color="#2c3e50", linewidth=1.5)
plt.plot(future_dates, future_pred, "o--", color="#e74c3c",
         linewidth=2, markersize=6, label=f"预测 (3个月)")
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

## 9. 加载已保存的状态

```python
# 恢复整个运行时
runtime2 = DeploymentRuntime.load("/tmp/dengue_runtime/")
print(f"加载: {runtime2}")
print(f"  vault: {runtime2.vault}")
print(f"  data_table: {len(runtime2.data_table)} 行")
print(f"  已 feed: {runtime2._feed_count} 次")

# 继续 feed
new_row = pd.DataFrame({"time": [(pd.Timestamp.now() + pd.DateOffset(months=1)).strftime("%Y-%m-%d")],
                        "cases": [np.random.randint(500, 1500)]})
result = runtime2.feed(new_row)
for name, r in result.items():
    if "error" not in r:
        print(f"  继续预测 — {name}: {r['pred'][:3].round(0).astype(int)}")
```

---

## 10. 附录

```python
print("可用模型:")
print(f"  Torch:   {list_models('torch')}")
print(f"  Sklearn: {list_models('sklearn')}")
print(f"  TimeSeries: {list_models('ts')}")
```

安装依赖：

```bash
pip install -e .                                # 基础
pip install torch                               # 深度学习
pip install -e ".[xgb,lgbm]"                   # 额外机器学习
pip install -e ".[all]"                        # 全部
```
