"""
预警评估模块 — 纯二分类预警评估。

核心：用 ``compare()`` 产出的测试集预测值直接做二分类预警。
- 预测值超过分位数 → 预警
- 实际值超过分位数 → 事件
- 不需要 RiskScorer，直接用 y_pred vs y_true 的错位矩阵

使用方法::

    from EpiAI.evaluate import warning_report

    task = ForecastTask(data=df, target="cases", freq="MS", horizon=12)
    vault = task.compare(["sf_auto_ets", "sf_holt_winters"])

    report = warning_report(
        vault=vault, bundle=task.build_bundle(), task=task,
        alert_percentile=0.90,   # 预测 > 历史 90% 分位 → 预警
        event_percentile=0.90,   # 实际 > 历史 90% 分位 → 事件
    )
    print(report)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def warning_report(
    vault,
    bundle,
    task,
    alert_percentile: float = 0.90,
    event_percentile: float = 0.90,
) -> pd.DataFrame:
    """利用 compare() 测试集预测值做二分类预警评估。

    对 vault 中每个模型、每个测试时间点：
    - 若 ``y_pred > alert_threshold`` → 预警
    - 若 ``y_true > event_threshold``  → 事件
    计算混淆矩阵 + 指标。

    Parameters
    ----------
    vault : ModelVault
        ``task.compare()`` 返回值。
    bundle : PipelineBundle
        ``task.build_bundle()`` 返回值。
    task : ForecastTask
        当前预测任务。
    alert_percentile : float, default 0.90
        预警阈值分位数。预测值超过此分位 → 预警。
    event_percentile : float, default 0.90
        事件阈值分位数。实际值超过此分位 → 事件。

    Returns
    -------
    pd.DataFrame
        ``model, auc, hit_rate, fa_rate, precision, recall, f1,
        n_alerts, n_events, tp, fp, fn, tn``
    """
    # 训练数据（train + val）
    train_data = _get_train_data(bundle, task)
    target = task.target[0]

    # 拟合期分位数阈值（基于训练数据）
    alert_threshold = float(train_data[target].quantile(alert_percentile))
    event_threshold = float(train_data[target].quantile(event_percentile))

    records: list[dict] = []
    test_predictions = getattr(vault, "test_predictions", {})

    if not test_predictions:
        raise ValueError("vault.test_predictions 为空，请确认 compare() 返回了测试集预测。")

    for model_name, td in test_predictions.items():
        y_pred = td["y_pred"].values
        y_true = td["y_true"].values
        time_col = task.time_col

        for h in range(len(y_pred)):
            is_warning = bool(y_pred[h] > alert_threshold)
            is_event = bool(y_true[h] > event_threshold)
            records.append({
                "model": model_name,
                "pred_value": float(y_pred[h]),
                "actual_value": float(y_true[h]),
                "is_warning": is_warning,
                "is_event": is_event,
            })

    if not records:
        raise RuntimeError("无评估记录生成。")

    result_df = pd.DataFrame(records)

    # ── 逐模型指标 ───────────────────────────────────────────
    metrics = []
    for model_name, grp in result_df.groupby("model"):
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

        # AUC: y_pred（连续值）vs is_event（二分类）
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(event, grp["pred_value"])
        except Exception:
            auc = np.nan

        metrics.append({
            "model": model_name,
            "auc": auc,
            "hit_rate": hit_rate,
            "fa_rate": fa_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_alerts": int(warned.sum()),
            "n_events": int(event.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return pd.DataFrame(metrics).sort_values("auc", ascending=False)


# ── 辅助 ──────────────────────────────────────────────────


def _get_train_data(bundle, task):
    train = bundle.train_df.copy()
    if bundle.val_df is not None:
        train = pd.concat([train, bundle.val_df], ignore_index=True)
    return train
