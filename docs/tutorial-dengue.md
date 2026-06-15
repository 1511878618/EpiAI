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

from EpiAI.models import sklearn_models, ts_models
from EpiAI.models import torch_models          # 需要 PyTorch，可选

from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, SlidingWindow,
)
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.inference import InferencePipeline, ModelVault
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

plt.figure(figsize=(14, 4))
plt.plot(df["time"], df["cases"], color="#2c3e50")
plt.title("全国登革热月发病数 (2010–2026)", fontsize=13)
plt.ylabel("病例数"); plt.grid(alpha=0.3); plt.show()
```

---

## 3. 数据管道

```python
df.to_csv("/tmp/dengue.csv", index=False)

bundle = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols="cases",
                     feature_cols="cases"),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=None,
    window=SlidingWindow(lookback=12, horizon=3),
).run("/tmp/dengue.csv")

print(f"训练: {bundle.train_x.shape}  验证: {bundle.val_x.shape}  测试: {bundle.test_x.shape}")
```

---

## 4. 训练所有模型

### 4.1 深度学习（需 PyTorch）

```python
results = {}

for name in ["MLP", "LSTM", "CNN"]:
    try:
        model = get(name)(input_dim=bundle.n_features, lookback=12, horizon=3, target_dim=1)
        r = EpiAITrainer(model=model, verbose=False, optimizer_config={"max_epochs": 10}).fit(bundle)
        results[name] = r
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")
```

### 4.2 机器学习

```python
for name, kwargs in [
    ("RF", {"n_estimators": 200, "max_depth": 10, "random_state": 42}),
    ("XGB", {"n_estimators": 200, "random_state": 42}),
    ("SVR", {"kernel": "rbf", "C": 1.0}),
]:
    try:
        model = get(name)(input_dim=bundle.n_features, lookback=12, horizon=3, target_dim=1,
                          **{f"{'rf' if name=='RF' else 'xgb' if name=='XGB' else 'svm'}_params": kwargs})
        results[name] = EpiAITrainer(model=model, verbose=False).fit(bundle)
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")
```

### 4.3 时间序列

```python
for name, kwargs in [("ETS", {"seasonal_periods": 12, "seasonal": "add", "trend": "add"}),
                      ("ARIMA", {"seasonal": True, "m": 12})]:
    try:
        model = get(name)(**kwargs)
        results[name] = EpiAITrainer(model=model, verbose=False).fit(bundle)
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")
```

---

## 5. ModelVault：入库、对比、部署

```python
vault = ModelVault.from_results(results, bundle)
vault.save("/tmp/dengue_vault/")

print("模型对比总表：")
print(vault.summary().to_string())
```

---

## 6. 全部模型预测可视化

```python
y_test = bundle.get_y_series("test")
n = bundle.get_y_series("test").shape[0]

plt.figure(figsize=(14, 6))
plt.plot(y_test, "o-", label="实际值", color="black", alpha=0.7, linewidth=2)

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for (name, r), color in zip(results.items(), colors):
    preds = r.predictions[:, -1, 0]
    m = r.metrics.iloc[0]
    plt.plot(preds, "s--", label=f"{name}  (R²={m['R2']:.3f})", color=color, alpha=0.6)

plt.legend(fontsize=9, ncol=3)
plt.title("登革热月发病数 — 多模型预测对比", fontsize=14)
plt.ylabel("病例数"); plt.xlabel("测试样本序号"); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/dengue_all_models.png", dpi=150)
plt.show()
```

---

## 7. 模型部署与实时更新

```python
vault = ModelVault.load("/tmp/dengue_vault/")

# ── 场景 1：所有模型对新数据做预测 ─────────────────────────
# 窗口模型需要特征列，TS 模型自动走 forecast
new_data = bundle.train_df.tail(15).copy()[bundle.feature_names]

all_preds = vault.predict_all(new_data=new_data, steps=6)
print("所有模型推理结果：")
for name, pred in all_preds.items():
    p = vault.get(name).paradigm
    val = pred[0, 0, 0] if p != "ts" else pred[0, 0, 0]
    print(f"  {name:8s} ({p:7s})  → 预测={val:.0f}  形状={pred.shape}")

# ── 场景 2：新观测值到达，更新时序模型 ─────────────────────
# 假设 6 月实际病例数为 1200，7 月为 800
new_cases = np.array([1200, 800], dtype=np.float32)

for name in ["ETS", "ARIMA"]:
    try:
        inferer = vault.get(name)
        updated = inferer.update(new_cases)
        print(f"  {name:8s} update → {updated.ravel().round(0).astype(int)}")
    except KeyError:
        pass  # 模型不在 vault 中（依赖未安装）

# ── 场景 3：从 vault 中挑选任一模型单独推理 ────────────────
best_name = vault.best("R2")
print(f"\n最佳模型单独部署: {best_name}")

inferer = vault.get(best_name)
if inferer.paradigm == "ts":
    fc = inferer.forecast(6)
    print(f"  forecast(6): {fc.ravel().round(0).astype(int)}")
else:
    pred = inferer.predict(new_data)
    print(f"  predict(15行 → 未来3月): {pred[0, :, 0].round(0).astype(int)}")

# 保存 vault（增量更新后）
vault.save("/tmp/dengue_vault/")
```

---

## 8. 附录

```python
print("可用模型:", list_models("torch"), list_models("sklearn"), list_models("ts"))
```
