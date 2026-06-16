# EpiAI 模型部署 —— 网站后端接口说明

> 面向网站开发人员。你不需要了解模型训练细节，只需知道如何加载模型、
> 传入数据、获取预测结果。

---

## 1. 整体流程

```
你的网站 / 数据库
    │
    ├─ 每月 1 次：上传新数据（CSV / JSON）
    │
    ▼
后端 Python 服务
    │
    ├─ DeploymentRuntime.feed(new_data)
    │    ① 检查时间是否连续
    │    ② 更新历史数据表
    │    ③ 运行所有模型进行预测
    │
    ▼
返回结果（JSON）
    {
      "RF":  {"time": ["2026-05", "2026-06", "2026-07"], "pred": [1043, 1144, 1376]},
      "ETS": {"time": ["2026-05", "2026-06", "2026-07"], "pred": [1106, 1365, 1016]}
    }
```

---

## 2. 需要的环境

```bash
# 后端服务器安装
pip install EpiAI

# 如果使用 XGBoost / LightGBM / ARIMA 等模型
pip install "EpiAI[xgb,lgbm,ts]"

# 或全部安装
pip install "EpiAI[all]"
```

---

## 3. 启动服务（初始化，只做一次）

```python
from EpiAI import ModelVault, DeploymentRuntime
import pandas as pd

# 加载训练好的模型库
vault = ModelVault.load("/path/to/model_vault/")

# 创建运行时
runtime = DeploymentRuntime(
    vault=vault,
    time_col="time",
    time_unit="MS",        # MS = 月
)

# 加载历史数据（可选，但建议加载以便窗口模型有足够历史）
runtime.data_table = pd.read_csv("/path/to/historical_data.csv")
runtime._train_end_time = pd.to_datetime("2026-03-01")  # 训练数据最后时间
```

---

## 4. 每月接收数据并返回预测（你的 API 接口）

```python
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# 注意：runtime 在上一步已经初始化好
# 实际部署时建议用全局变量或单例模式管理

@app.route("/predict", methods=["POST"])
def predict():
    """
    接收新数据，返回所有模型的预测结果。

    请求体 (JSON):
    {
        "time": "2026-04-01",       # 数据对应的时间
        "cases": 1500               # 实际病例数（或其他特征列）
    }

    响应 (JSON):
    {
        "RF":  {"time": ["2026-05", "2026-06", "2026-07"], "pred": [983, 1043, 1144]},
        "ETS": {"time": ["2026-05", "2026-06", "2026-07"], "pred": [1106, 1365, 1016]}
    }
    """
    data = request.get_json()

    # 转为 DataFrame
    new_row = pd.DataFrame([data])

    # feed 给运行时 → 自动检查时间连续性 + 预测
    result = runtime.feed(new_row)

    # 整理为可序列化的格式
    output = {}
    for name, r in result.items():
        if "error" in r:
            output[name] = {"error": r["error"]}
        else:
            output[name] = {
                "time": [t.strftime("%Y-%m") for t in r["time"]],
                "pred": [round(float(v)) for v in r["pred"]],
            }

    return jsonify(output)
```

---

## 5. 前端调用示例

```javascript
// 每月发一次请求
fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        time: "2026-04-01",
        cases: 1500
    })
})
.then(res => res.json())
.then(data => {
    console.log("RF 预测:", data.RF);
    console.log("ETS 预测:", data.ETS);

    // 示例：在页面上展示最佳模型的预测
    const rf = data.RF;
    document.getElementById("next_month").innerText = rf.pred[0];
    document.getElementById("chart").dataset.values = JSON.stringify(rf.pred);
});
```

---

## 6. 时间要求

| 规则 | 说明 |
|------|------|
| **每月调用一次** | 每次 POST 包含一个月的数据 |
| **时间必须连续** | 上个月是 `2026-03`，下个月必须是 `2026-04` |
| **不能跳过** | `2026-03` 后传 `2026-05` 会报错（缺口） |
| **不能重复** | 已传过 `2026-04` 再传一次会报错 |
| **可关闭严格模式** | 初始化时设置 `strict=False`，缺口仅警告不报错 |

---

## 7. 返回结果说明

```json
{
  "RF": {
    "time": ["2026-05", "2026-06", "2026-07"],   // 未来 3 个月的时间标签
    "pred": [983, 1043, 1144]                     // 对应的预测值
  },
  "ETS": {
    "time": ["2026-05", "2026-06", "2026-07"],
    "pred": [1106, 1365, 1016]
  }
}
```

- `pred[0]` 是**下一个月的预测值**
- `pred[1]` 是**下两个月的预测值**
- 以此类推，共 `k` 个（k = horizon，训练时设定，通常为 3 或 6）

---

## 8. 完整示例文件

见 `docs/deployment-example.md`（含更多场景）。

`tutorial/tutorial-full.ipynb` 中有完整的训练→部署模拟流程，第 7 节演示了逐月 feed 的过程。
