"""
预警规则模块：组合多模型风险等级，生成预警报告。

支持三种融合策略：:

    - ``max``      : 取所有模型中的最高风险（偏保守）
    - ``mean``     : 取平均风险（偏平滑）
    - ``consensus``: 至少 N 个模型同意才升级（避免单模型误报）
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# ── 风险等级标签 ─────────────────────────────────────────────

RISK_LABELS = {0: "低", 1: "中", 2: "高", 3: "极高"}
RISK_COLORS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
RISK_ACTIONS = {
    0: "常规监测",
    1: "关注趋势变化",
    2: "建议加强监测",
    3: "⚠ 建议启动应急响应",
}


class WarningRule:
    """组合多模型风险等级，生成预警报告。

    Parameters
    ----------
    ensemble : str
        融合策略: ``"max"`` / ``"mean"`` / ``"consensus"``
    min_agreement : int, optional
        ``consensus`` 模式下至少需要多少个模型同意当前等级。
    escalation_months : int, default=2
        连续多少个月高风险触发升级提醒。
    show_detail : bool, default=True
        输出报告中是否包含各模型分项风险。
    """

    def __init__(
        self,
        ensemble: str = "max",
        min_agreement: int | None = None,
        escalation_months: int = 2,
        show_detail: bool = True,
    ):
        self.ensemble = ensemble
        self.min_agreement = min_agreement
        self.escalation_months = escalation_months
        self.show_detail = show_detail

    def evaluate(
        self,
        risk_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """评估风险表，生成含融合风险 + 预警建议的报告。

        Parameters
        ----------
        risk_df : pd.DataFrame
            ``RiskScorer.score_df()`` 的输出，行=时间，列=模型，值为 0-3 整数。

        Returns
        -------
        pd.DataFrame
            含以下列的预警报告:
            - 各模型风险（如果 show_detail=True）
            - 综合风险: 融合后的风险等级
            - 预警动作: 对应行动建议
            - 升级标记: 连续高风险提醒
        """
        result = risk_df.copy()

        # ── 融合多模型风险 ─────────────────────────────────
        result["综合风险"] = self._aggregate(risk_df.values)

        # ── 预警动作 ────────────────────────────────────────
        result["预警动作"] = result["综合风险"].map(RISK_ACTIONS)

        # ── 升级标记（连续 N 个月高风险） ────────────────────
        high_series = (result["综合风险"] >= 2).astype(int)
        streak = high_series.groupby(
            (high_series != high_series.shift()).cumsum()
        ).cumsum()
        result["升级标记"] = ""
        for i in range(self.escalation_months - 1, len(streak)):
            if streak.iloc[i] >= self.escalation_months:
                prev = result["综合风险"].iloc[i]
                label = RISK_LABELS.get(prev, str(prev))
                result["升级标记"].iloc[i] = (
                    f"连续{self.escalation_months}个月{label}风险，请注意"
                )

        return result

    def _aggregate(self, risk_array: np.ndarray) -> np.ndarray:
        """融合多模型风险等级。"""
        if self.ensemble == "max":
            return risk_array.max(axis=1)
        elif self.ensemble == "mean":
            return np.round(risk_array.mean(axis=1)).astype(int)
        elif self.ensemble == "consensus":
            n_models = risk_array.shape[1]
            min_agr = self.min_agreement or max(2, n_models // 2)
            result = []
            for row in risk_array:
                # 统计每个等级有多少模型
                levels, counts = np.unique(row, return_counts=True)
                max_count = counts.max()
                max_level = levels[counts.argmax()]
                if max_count >= min_agr:
                    result.append(max_level)
                else:
                    # 没有足够模型达成一致，取最高风险
                    result.append(row.max())
            return np.array(result, dtype=int)
        else:
            raise ValueError(f"Unknown ensemble: {self.ensemble}")

    def report(self, result: pd.DataFrame) -> str:
        """生成可读的预警报告文本。

        Parameters
        ----------
        result : pd.DataFrame
            ``evaluate()`` 的输出。

        Returns
        -------
        str
            格式化报告文本。
        """
        lines = ["=" * 50, "  传染病风险预警报告", "=" * 50, ""]

        for idx, row in result.iterrows():
            risk = int(row["综合风险"])
            color = RISK_COLORS.get(risk, "⚪")
            label = RISK_LABELS.get(risk, "?")
            action = row["预警动作"]
            mark = row.get("升级标记", "")

            if self.show_detail:
                model_risks = ", ".join(
                    f"{col}:{RISK_LABELS.get(int(row[col]), '?')}"
                    for col in result.columns
                    if col not in ("综合风险", "预警动作", "升级标记")
                )
                lines.append(f"  {idx}  {color} {label}风险  |  {model_risks}")
            else:
                lines.append(f"  {idx}  {color} {label}风险")

            lines.append(f"        → {action}")
            if mark:
                lines.append(f"        ⚠ {mark}")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)
