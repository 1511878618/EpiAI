"""
风险评分模块：将预测值映射为风险等级。

支持四种评分方法：

- ``quantile``: 基于历史同期分布的分位数
- ``threshold``: 基于绝对阈值
- ``zscore``: 基于与历史均值的偏离
- ``pct_change``: 基于环比/同比变化率

风险等级::

    0 = 低风险 (正常范围)
    1 = 中风险 (关注)
    2 = 高风险 (警告)
    3 = 极高风险 (警报)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class RiskScorer:
    """将预测值转为风险等级。

    Parameters
    ----------
    method : str
        评分方法: ``"quantile"`` / ``"threshold"`` / ``"zscore"`` / ``"pct_change"``
    time_col : str
        历史数据的时间列名。
    value_col : str
        历史数据的数值列名（用于计算基线）。
    thresholds : list of float, optional
        ``method="threshold"`` 时的绝对阈值，长度为 3：[中, 高, 极高]。
        例如 ``[500, 1000, 2000]``。
    quantile_bounds : list of float, optional
        ``method="quantile"`` 时的分位点，长度为 3：[中, 高, 极高]。
        默认 ``[0.7, 0.9, 0.95]``。
    zscore_bounds : list of float, optional
        ``method="zscore"`` 时的 z-score 阈值，长度为 3。
        默认 ``[1.0, 2.0, 3.0]``。
    pct_bounds : list of float, optional
        ``method="pct_change"`` 时的环比变化率阈值，长度为 3。
        默认 ``[0.3, 0.5, 1.0]`` (30%, 50%, 100%)。
    lookback : int, optional
        ``pct_change`` 方法计算环比的窗口大小，默认 1（环比上月）。
    same_period : bool, default=True
        计算分位数/z-score 时是否只使用历史同期数据
        （如预测 6 月，则只参考历史所有 6 月的数据）。
    """

    def __init__(
        self,
        method: str = "quantile",
        time_col: str = "time",
        value_col: str = "cases",
        thresholds: list[float] | None = None,
        quantile_bounds: list[float] | None = None,
        zscore_bounds: list[float] | None = None,
        pct_bounds: list[float] | None = None,
        lookback: int = 1,
        same_period: bool = True,
    ):
        self.method = method
        self.time_col = time_col
        self.value_col = value_col
        self.lookback = lookback
        self.same_period = same_period

        self._bounds = {
            "quantile": quantile_bounds or [0.70, 0.90, 0.95],
            "threshold": thresholds or [500, 1000, 2000],
            "zscore": zscore_bounds or [1.0, 2.0, 3.0],
            "pct_change": pct_bounds or [0.30, 0.50, 1.00],
        }

        # 存储历史基线（由 fit() 计算）
        self._baseline: dict[str, np.ndarray] = {}

    def fit(self, history: pd.DataFrame) -> "RiskScorer":
        """基于历史数据计算基线分布。

        Parameters
        ----------
        history : pd.DataFrame
            历史数据表（即 ``data_table``），必须含 ``time_col`` 和 ``value_col``。
        """
        if self.method == "quantile":
            self._fit_quantile(history)
        elif self.method == "zscore":
            self._fit_zscore(history)
        return self

    def _fit_quantile(self, history: pd.DataFrame):
        """计算各月份的历史分位数。"""
        vals = pd.to_datetime(history[self.time_col])
        months = vals.dt.month
        bounds = self._bounds["quantile"]
        for month in range(1, 13):
            mask = months == month
            subset = history.loc[mask, self.value_col].values
            if len(subset) < 3:
                # 数据不足时用全部历史
                subset = history[self.value_col].values
            q = np.quantile(subset, bounds)
            self._baseline[month] = q

    def _fit_zscore(self, history: pd.DataFrame):
        """计算各月份的历史均值和标准差。"""
        vals = pd.to_datetime(history[self.time_col])
        months = vals.dt.month
        for month in range(1, 13):
            mask = months == month
            subset = history.loc[mask, self.value_col].values
            if len(subset) < 3:
                subset = history[self.value_col].values
            self._baseline[month] = np.array([np.nanmean(subset), np.nanstd(subset) or 1.0])

    # ── 风险评分 ─────────────────────────────────────────────

    def score(self, value: float, time: pd.Timestamp) -> int:
        """对单个值进行风险评分。

        Parameters
        ----------
        value : float
            预测值。
        time : pd.Timestamp
            该预测对应的时间点。

        Returns
        -------
        int
            风险等级 0-3。
        """
        if self.method == "quantile":
            return self._score_quantile(value, time)
        elif self.method == "threshold":
            return self._score_threshold(value)
        elif self.method == "zscore":
            return self._score_zscore(value, time)
        elif self.method == "pct_change":
            return self._score_pct(value)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _score_quantile(self, value: float, time: pd.Timestamp) -> int:
        month = time.month
        q = self._baseline.get(month, np.array([float("inf")]) * 3)
        if value >= q[2]:
            return 3  # >95% 极高
        elif value >= q[1]:
            return 2  # 90-95% 高
        elif value >= q[0]:
            return 1  # 70-90% 中
        return 0  # <70% 低

    def _score_threshold(self, value: float) -> int:
        t = self._bounds["threshold"]
        if value >= t[2]:
            return 3
        elif value >= t[1]:
            return 2
        elif value >= t[0]:
            return 1
        return 0

    def _score_zscore(self, value: float, time: pd.Timestamp) -> int:
        month = time.month
        mean_, std_ = self._baseline.get(month, [0, 1])
        z = (value - mean_) / std_
        b = self._bounds["zscore"]
        if z >= b[2]:
            return 3
        elif z >= b[1]:
            return 2
        elif z >= b[0]:
            return 1
        return 0

    def _score_pct(self, value: float) -> int:
        # 环比变化率需要历史值，在此简化为 threshold 模式
        # 实际使用时应通过 score_df 传入 prev_value
        return self._score_threshold(value)

    # ── 批量评分 ─────────────────────────────────────────────

    def score_df(
        self,
        predictions: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """对预测结果表进行批量风险评分。

        Parameters
        ----------
        predictions : pd.DataFrame
            ``runtime.predict(horizon=N)`` 返回的 DataFrame。
            行 = 时间（字符串 "YYYY-MM"），列 = 模型名。
        history : pd.DataFrame, optional
            历史数据。如果未传入，使用 ``fit()`` 时缓存的数据。

        Returns
        -------
        pd.DataFrame
            行 = 时间，每模型一列风险等级值的 DataFrame。
        """
        result = predictions.copy()
        times = pd.to_datetime(result.index)

        if self.method in ("pct_change",):
            # 环比方法需要上一步的预测值
            prev_vals = {col: np.nan for col in predictions.columns}
            for i, (idx, row) in enumerate(predictions.iterrows()):
                t = times[i]
                for col in predictions.columns:
                    val = row[col]
                    prev = prev_vals[col]
                    # 用当前值 vs 历史最后值 计算变化率
                    result.loc[idx, col] = self._rate_to_level(val, prev)
                    prev_vals[col] = val
        else:
            for i, idx in enumerate(result.index):
                t = times[i]
                for col in result.columns:
                    val = result.loc[idx, col]
                    result.loc[idx, col] = self.score(float(val), t)

        return result.astype(int)

    def _rate_to_level(self, val: float, prev: float | None) -> int:
        """将变化率映射到风险等级。"""
        if prev is None or np.isnan(prev) or prev <= 0:
            return 0
        rate = (val - prev) / prev
        b = self._bounds["pct_change"]
        if rate >= b[2]:
            return 3
        elif rate >= b[1]:
            return 2
        elif rate >= b[0]:
            return 1
        return 0
