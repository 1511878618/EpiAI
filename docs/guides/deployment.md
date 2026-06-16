# 部署模块指南

## 功能介绍

部署模块位于 `EpiAI.inference`，包含三个层次：单模型推理、多模型管理、生产部署。

### 三层结构

```
InferencePipeline  →  单模型推理 / 持久化
ModelVault         →  多模型存储 / 对比 / 批量推理
DeploymentRuntime  →  生产部署 / 时间检查 / 持久化
```

---

### InferencePipeline

封装一个已训练的模型和变换管道。

```python
from EpiAI import InferencePipeline

# 从训练结果构建
inferer = InferencePipeline.from_train_result(result)
inferer.save("/path/to/model.zip")

# 恢复
inferer = InferencePipeline.load("/path/to/model.zip")

# 推理
pred = inferer.predict(new_data_df)           # 窗口模型
forecast = inferer.forecast(6)                # TS 模型
updated = inferer.update(np.array([1200]))    # TS 在线更新
```

### ModelVault

管理和对比多个训练好的模型。

```python
from EpiAI import ModelVault

vault = ModelVault.from_results({"RF": r_rf, "XGB": r_xgb}, bundle)
vault.save("/tmp/vault/")

# 对比表
print(vault.summary())

# 选最优
best_name = vault.best("R2")

# 批量推理
results = vault.predict_all(new_data)

# 单模型取用
inferer = vault["RF"]
```

### DeploymentRuntime

生产部署运行时，维护统一历史数据表。

```python
from EpiAI import DeploymentRuntime

runtime = DeploymentRuntime(vault, time_col="time", time_unit="MS")
runtime.data_table = historical_data.copy()

# 每月调用
result = runtime.feed(new_observation)
```

详见 [部署 API 参考](../api-deployment.md)。

---

## 数据流

```
训练阶段：
  CSV → ForecastPipeline → bundle → EpiAITrainer.fit() → TrainResult
  TrainResult → InferencePipeline.save("model.zip")     ← 单模型
  TrainResult → ModelVault.save("/path/to/vault/")      ← 多模型

部署阶段：
  ModelVault.load() → DeploymentRuntime → runtime.feed(new_data)
                                        → {模型名: {time, pred}}
```

---

## 持久化格式

| 方式 | 格式 | 内容 |
|------|------|------|
| InferencePipeline | `model.zip` | config.json + model.pkl + transforms.pkl |
| ModelVault | `目录` | manifest.json + 每个模型一个子目录 |
| DeploymentRuntime | `目录` | runtime_meta.json + data_table.parquet + vault/ |

---

## 扩展指南

### 添加新的持久化后端

默认使用 pickle 序列化模型。如需替换为其他格式（如 torch.jit、ONNX），重写 `BaseForecaster.save()` / `load()`：

```python
class MyModel(SomeMixin):
    def save(self, path):
        torch.save(self.state_dict(), path)
        json.dump({"config": self.config}, open(path + ".cfg", "w"))

    @classmethod
    def load(cls, path):
        config = json.load(open(path + ".cfg"))
        model = cls(**config)
        model.load_state_dict(torch.load(path))
        return model
```

### 添加新的持久化后端到 InferencePipeline

修改 `InferencePipeline.save()` 中的模型序列化部分：

```python
def save(self, path):
    zip_path = Path(path).with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("config.json", json.dumps(config))
        zf.writestr("model.pkl", pickle.dumps(self.model))  # ← 替换此行
        zf.writestr("transforms.pkl", pickle.dumps(self.transforms))
```

### 自定义部署逻辑

`DeploymentRuntime.feed()` 内部的模型循环可以覆写或扩展。例如在 feed 前后加入日志、告警等：

```python
class LoggingRuntime(DeploymentRuntime):
    def feed(self, new_data):
        print(f"[{datetime.now()}] feed {len(new_data)} rows")
        result = super().feed(new_data)
        for name, r in result.items():
            if "error" in r:
                print(f"[WARN] {name}: {r['error']}")
        return result
```
