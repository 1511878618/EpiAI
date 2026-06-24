"""风险预警模块：将预测值转为风险等级并生成预警报告。

快速开始::

    from EpiAI.risk import RiskScorer, WarningRule

    # 1. 评分器：用历史数据拟合分位数基线
    scorer = RiskScorer(method="quantile").fit(runtime.data_table)

    # 2. 对预测结果进行风险评分
    preds = runtime.predict(horizon=12)  # pd.DataFrame
    risk_df = scorer.score_df(preds)

    # 3. 融合多模型 + 生成预警报告
    rule = WarningRule(ensemble="max")
    report = rule.evaluate(risk_df)
    print(rule.report(report))
"""

from .scorer import RiskScorer
from .rules import WarningRule, RISK_LABELS, RISK_COLORS, RISK_ACTIONS

__all__ = [
    "RiskScorer",
    "WarningRule",
    "RISK_LABELS",
    "RISK_COLORS",
    "RISK_ACTIONS",
]
