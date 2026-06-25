# 风险预警模块指南

## 功能介绍

风险预警模块位于 `EpiAI.risk`，是一个完全独立于框架其他部分的模块。
它的职责是将**预测值**转化为**风险等级**和**可执行的预警建议**。

### 核心设计

```
预测值 (原始尺度)             风险等级 (0-3)            预警报告
    │                            │                        │
    ▼                            ▼                        ▼
RiskScorer                  WarningRule               report()
score_df(preds)         →  evaluate(risk_df)    →  文本/DataFrame
```

### 风险等级

| 等级 | 标签 | 行动建议 |
|------|------|---------|
| 0 | 低风险 🟢 | 常规监测 |
| 1 | 中风险 🟡 | 关注趋势变化 |
| 2 | 高风险 🟠 | 建议加强监测 |
| 3 | 极高风险 🔴 | ⚠ 建议启动应急响应 |

---

## RiskScorer

将预测值映射为风险等级。

### 初始化

```python
from EpiAI.risk import RiskScorer

scorer = RiskScorer(
    method="quantile",          # 评分方法
    time_col="time",            # 时间列名
    value_col="cases",          # 数值列名
    same_period=True,           # 分位数是否按同期（如6月vs历史6月）
)
```

### 拟合基线

```python
scorer = scorer.fit(history_df)
# 对 quantile 方法：计算每个月份的历史分位数（70%/90%/95%）
# 对 zscore 方法：计算每个月份的历史均值和标准差
```

### 批量评分

```python
preds = runtime.predict(horizon=12)     # pd.DataFrame
risk_df = scorer.score_df(preds)
# 输出: 与 preds 相同结构，值变为 0-3 整数
```

### 评分方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `quantile` | `quantile_bounds=[0.70, 0.90, 0.95]` | 历史同期分位数（默认） |
| `threshold` | `thresholds=[500, 1000, 2000]` | 绝对阈值 |
| `zscore` | `zscore_bounds=[1.0, 2.0, 3.0]` | 偏离历史均值的标准差数 |
| `pct_change` | `pct_bounds=[0.30, 0.50, 1.00]` | 环比变化率 |

---

## WarningRule

组合多模型风险等级，生成预警报告。

### 初始化

```python
from EpiAI.risk import WarningRule

rule = WarningRule(
    ensemble="max",              # 融合策略
    escalation_months=2,         # 连续几个月高风险触发升级提醒
)
```

### 评估

```python
report_df = rule.evaluate(risk_df)
# 输出: 含 [综合风险] 和 [预警动作] 列的 DataFrame
```

### 文本报告

```python
print(rule.report(report_df))
# 输出:
# ==================================================
#   传染病风险预警报告
# ==================================================
#   2026-05  🟢 低风险  |  RF:低, ARIMA:低, Prophet:低
#         → 常规监测
#
#   2026-08  🔴 极高风险  |  RF:极高, ARIMA:高, Prophet:极高
#         → ⚠ 建议启动应急响应
# ==================================================
```

### 融合策略

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| `max` | 取所有模型中的最高风险 | 保守预警，不漏报 |
| `mean` | 取平均风险（取整） | 平滑输出，减少误报 |
| `consensus` | 至少 N 个模型同意 | 需要多数一致才预警 |

---

## 完整示例

```python
from EpiAI.risk import RiskScorer, WarningRule
from EpiAI.inference import DeploymentRuntime

# 1. 设置评分器
scorer = RiskScorer(
    method="quantile",
    value_col="cases",
).fit(runtime.data_table)

# 2. 评分
preds = runtime.predict(horizon=12)
risk_df = scorer.score_df(preds)

# 3. 预警
rule = WarningRule(ensemble="max", escalation_months=2)
report = rule.evaluate(risk_df)
print(rule.report(report))

# 4. 导出 CSV
report.to_csv("warning_report.csv")
```

---

## 与 DeploymentRuntime 配合

```python
runtime = DeploymentRuntime(vault=vault, data_table=history_df)
runtime.update_ts()
preds = runtime.predict(horizon=12)

# 风险预警（完全独立，不依赖 runtime 状态）
scorer = RiskScorer(method="quantile").fit(runtime.data_table)
rule = WarningRule(ensemble="max", escalation_months=3)
report = rule.evaluate(scorer.score_df(preds))
```
