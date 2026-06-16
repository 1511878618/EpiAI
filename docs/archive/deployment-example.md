# EpiAI 模型部署样例

> 面向对接方。以一个已训练好的登革热模型为例，展示如何加载模型、
> 对新数据做预测、以及持续接入实时数据。

---

## 1. 项目结构

```
EpiAI-dev/
├── tutorial/
│   └── tutorial-full.ipynb    ← 完整训练+部署流程
│
├── docs/
│   └── deployment-design.md   ← 部署设计文档
│
└── src/EpiAI/
    └── inference.py           ← InferencePipeline, ModelVault, DeploymentRuntime
```

---

## 2. 单次推理（最简单的用法）

给模型一份新数据，得到预测结果。

```python
from EpiAI import InferencePipeline

# 加载已训练好的模型
inferer = InferencePipeline.load("/path/to/model.zip")

# 准备新数据（至少需要 lookback 行）
new_data = pd.DataFrame({
    "time": ["2026-04-01", "2026-05-01", "2026-06-01"],
    "cases": [1500, 1200, 800],          # 特征列名取决于训练时的配置
})

# 预测未来 k 步
pred = inferer.predict(new_data)
# pred: (N, horizon, target_dim) — 例如 (1, 3, 1) 表示未来 3 个月的预测

print(pred[0, :, 0])  # 未来 3 个月的预测值
```

**输出示例：**
```
[1043. 1144. 1376.]   # 下一个月 = 1043，下下个月 = 1144，第三个月 = 1376
```

---

## 3. 多模型管理（对比多个模型）

训练好的多个模型统一打包，方便对比和选优。

```python
from EpiAI import ModelVault

# 加载模型库
vault = ModelVault.load("/path/to/vault/")

# 查看所有模型的指标对比
print(vault.summary())

# 选最优模型
best_name = vault.best("R2")      # 按 R² 选
inferer = vault.get(best_name)

# 所有模型同时预测
results = vault.predict_all(new_data)
for name, pred in results.items():
    print(f"{name}: {pred}")
```

**输出示例：**
```
         paradigm       MAE       RMSE       R2  PearsonR
model                                                     
RF       sklearn   758.97   1366.75   0.693     0.866
ETS           ts  1772.79   2681.61  -0.239    -0.252
```

---

## 4. 生产部署（持续接收数据）

模拟月度数据的持续接入。模型自动检查数据是否连续，产出预测。

```python
from EpiAI import DeploymentRuntime

# 初始化运行时
runtime = DeploymentRuntime(
    vault=vault,
    time_col="time",        # 时间列名
    time_unit="MS",         # 时间单位：月
)

# 载入历史数据（训练数据）
runtime.data_table = historical_data.copy()
runtime._train_end_time = historical_data["time"].iloc[-1]

# ── 每收到一条新数据，调用 feed() ──
new_observation = pd.DataFrame({
    "time": ["2026-04-01"],
    "cases": [1500],
})

result = runtime.feed(new_observation)

for name, pred in result.items():
    if "error" not in pred:
        print(f"{name}: {pred['time']} → {pred['pred']}")
```

**输出示例：**
```
RF:  ['2026-05', '2026-06', '2026-07'] → [ 983. 1043. 1144.]
ETS: ['2026-05', '2026-06', '2026-07'] → [1106. 1365. 1016.]
```

---

## 5. 时间连续性检查

`DeploymentRuntime` 自动检查新数据的时间是否正确，防止数据缺失或乱序。

| 情况 | 行为 |
|------|------|
| 上个月是 2026-03，新数据是 2026-04 | ✅ 正常 |
| 上个月是 2026-03，新数据是 2026-05 | ❌ TimeGapError（缺了 2026-04） |
| 上个月是 2026-03，新数据是 2026-03 | ❌ TimeOrderError（时间重复） |
| 不需要严格模式 | 设置 `strict=False`，缺口只警告不报错 |

---

## 6. 模型持久化

```python
# 保存单个模型
inferer.save("/models/dengue_rf.zip")
inferer = InferencePipeline.load("/models/dengue_rf.zip")

# 保存模型库
vault.save("/models/dengue_vault/")
vault = ModelVault.load("/models/dengue_vault/")

# 保存运行时（含全部历史数据 + 模型状态）
runtime.save("/models/dengue_runtime/")
runtime = DeploymentRuntime.load("/models/dengue_runtime/")
```

---

## 7. 快速运行完整样例

```bash
# 1. 安装
pip install -e .                      # 基础版
pip install -e ".[xgb,lgbm,ts]"      # 含 XGBoost、LightGBM、ARIMA、ETS

# 2. 运行教程
jupyter notebook tutorial/tutorial-full.ipynb
```

---

## 8. 关键文件说明

| 文件 | 用途 |
|------|------|
| `src/EpiAI/inference.py` | `InferencePipeline` — 单模型推理 |
| | `ModelVault` — 多模型管理 |
| | `DeploymentRuntime` — 生产部署 |
| `src/EpiAI/trainer.py` | `EpiAITrainer` — 模型训练 |
| `src/EpiAI/dataset/pipeline.py` | `ForecastPipeline` — 数据管道 |
