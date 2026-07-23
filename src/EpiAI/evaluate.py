"""
预警评估模块。

核心思路：``compare()`` 阶段已有测试集的 ``y_pred`` 和 ``y_true``，
直接利用它们来评估不同风险算法的预警准确率，不需要额外训练。

使用方法::

    from EpiAI.evaluate import compare_risk_methods

    task = ForecastTask(data=df, target="cases", freq="MS", horizon=12)
    bundle = task.build_bundle()
    vault = task.compare(["sf_auto_ets", "sf_holt_winters"])

    report = compare_risk_methods(
        vault=vault,
        bundle=bundle,
        task=task,
        risk_methods=["quantile", "zscore", "threshold"],
        event_threshold=0.90,         # 事件 = 实际值 > 历史 90% 分位
    )
    print(report)

之后 deploy 时就可以根据 report 选择最优的风险算法组合。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compare_risk_methods(
    vault,
    bundle,
    task,
    risk_methods: list[str] = ("quantile", "zscore", "threshold"),
    event_threshold: float | str = 0.90,
    warning_level: int = 2,
) -> pd.DataFrame:
    """在 ``compare()`` 产出的测试集上，评估多种风险算法的预警准确率。

    利用 ``vault.test_predictions`` 中的 ``y_true`` / ``y_pred``，
    分别在每种风险算法下计算混淆矩阵和预警指标。

    Parameters
    ----------
    vault : ModelVault
        ``task.compare()`` 的返回值。
    bundle : PipelineBundle
        ``task.build_bundle()`` 的返回值。
    task : ForecastTask
        当前的预测任务对象。
    risk_methods : list of str
        要评估的风险算法列表，默认 ``["quantile", "zscore", "threshold"]``。
    event_threshold : float or str
        事件定义：

        - ``0.90``：实际值 > 历史同期 90% 分位 → 事件
        - ``500``（数值）：实际值 > 500 → 事件
        - ``"2x_median"``：实际值 > 历史中位数 × 2
    warning_level : int
        预警触发等级，默认 ``2``（风险 ≥ 2 视为预警）。

    Returns
    -------
    pd.DataFrame
        每模型 × 每种风险算法的评估指标：

        ``model, risk_method, hit_rate, fa_rate, precision, recall, f1,
        n_alerts, n_events, tp, fp, fn, tn``
    """
    from EpiAI.risk import RiskScorer

    # ── 训练数据（用于拟合风险评分器的基线） ─────────────────
    train_data = _get_train_data(bundle, task)

    # ── 事件阈值（基于训练数据计算） ─────────────────────────
    actual_threshold = _resolve_threshold(
        train_data, task.target[0], task.time_col, event_threshold,
    )

    # ── 逐模型 × 逐风险算法评估 ─────────────────────────────
    records: list[dict] = []
    test_predictions = getattr(vault, "test_predictions", {})

    if not test_predictions:
        raise ValueError(
            "vault.test_predictions 为空。请确认 compare() 返回的 vault 包含测试集预测。"
        )

    for model_name, td in test_predictions.items():
        y_pred = td["y_pred"].values
        y_true = td["y_true"].values
        times = pd.to_datetime(td[task.time_col].values)

        for rm in risk_methods:
            scorer = RiskScorer(method=rm, time_col=task.time_col,
                                value_col=task.target[0], same_period=True)
            scorer.fit(train_data)

            for h in range(len(y_pred)):
                risk_level = scorer.score(float(y_pred[h]), times[h])
                is_warning = risk_level >= warning_level
                actual_val = y_true[h]
                is_event = bool(actual_val > actual_threshold)

                records.append({
                    "model": model_name,
                    "risk_method": rm,
                    "pred_time": times[h],
                    "pred_value": float(y_pred[h]),
                    "actual_value": float(actual_val),
                    "risk_level": int(risk_level),
                    "is_warning": is_warning,
                    "is_event": is_event,
                })

    if not records:
        raise RuntimeError("没有生成任何评估记录。")

    result_df = pd.DataFrame(records)

    # ── 计算评估指标 ─────────────────────────────────────────
    metrics = []
    for (model, rm), grp in result_df.groupby(["model", "risk_method"]):
        warned = grp["is_warning"]
        event = grp["is_event"]
        tp = int((warned & event).sum())
        fp = int((warned & ~event).sum())
        fn = int((~warned & event).sum())
        tn = int((~warned & ~event).sum())

        def _safe(a, b):
            return a / b if b > 0 else np.nan

        hit_rate = _safe(tp, tp + fn)
        fa_rate = _safe(fp, fp + tn)
        precision = _safe(tp, tp + fp)
        recall = hit_rate
        f1 = _safe(2 * precision * recall, precision + recall)

        metrics.append({
            "model": model, "risk_method": rm,
            "hit_rate": hit_rate, "fa_rate": fa_rate,
            "precision": precision, "recall": recall, "f1": f1,
            "n_alerts": int(warned.sum()),
            "n_events": int(event.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(metrics).sort_values(["model", "risk_method"])


# ── 阈值辅助 ────────────────────────────────────────────────


def _resolve_threshold(train_data, target, time_col, threshold):
    if isinstance(threshold, (int, float)):
        if 0 < threshold < 1:
            return float(train_data[target].quantile(threshold))
        return float(threshold)
    if isinstance(threshold, str):
        if threshold.startswith("p"):
            return float(train_data[target].quantile(float(threshold[1:]) / 100))
        if threshold == "2x_median":
            return float(train_data[target].median() * 2)
    raise ValueError(f"无法解析阈值: {threshold}")


def _get_train_data(bundle, task):
    """从 bundle 中拿到训练数据（train + val 合并）。"""
    train = bundle.train_df.copy()
    if bundle.val_df is not None:
        train = pd.concat([train, bundle.val_df], ignore_index=True)
    return train
