"""
预警评估模块 — 多方法二分类预警评估。

支持多种方法定义阈值，然后做纯二分类：
  - 预测值超过阈值 → 预警
  - 实际值超过阈值 → 事件

方法:
  - "zscore":    阈值 = mean + level × std
  - "quantile":  阈值 = level 分位数（如 0.90）
  - "threshold": 阈值 = level（绝对值）

使用方法::

    from EpiAI.evaluate import warning_report

    task = ForecastTask(data=df, target="cases", freq="MS", horizon=12)
    vault = task.compare(["sf_auto_ets", "sf_holt_winters"])

    # 用 zscore 定义预警和事件
    report = warning_report(vault, bundle, task,
                            alert_method="zscore",    alert_level=1.0,
                            event_method="zscore",    event_level=1.0)
    print(report)

    # 对比多种方法
    for method in ["quantile", "zscore", "threshold"]:
        r = warning_report(vault, bundle, task, method=method, level=0.90)
        print(r[["model", "auc", "hit_rate", "fa_rate", "f1"]])
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def warning_report(
    vault,
    bundle,
    task,
    method: str = "quantile",
    level: float = 0.90,
    alert_method: str | None = None,
    event_method: str | None = None,
    alert_level: float | None = None,
    event_level: float | None = None,
) -> pd.DataFrame:
    """利用 compare() 测试集预测值做二分类预警评估。

    支持多种阈值计算方法，每种方法只需一个 level 参数：
    - ``"quantile"``:  level=0.90 → 90% 分位数
    - ``"zscore"``:    level=2.0  → 均值 + 2×标准差
    - ``"threshold"``: level=500  → 绝对值 500

    预警和事件的阈值可分别指定（通过 alert_method / event_method），
    也可统一指定（通过 method / level）。

    Parameters
    ----------
    vault : ModelVault
        ``task.compare()`` 返回值。
    bundle : PipelineBundle
        ``task.build_bundle()`` 返回值。
    task : ForecastTask
        当前预测任务。
    method : str, default "quantile"
        统一指定预警和事件的阈值计算方法。
    level : float, default 0.90
        统一指定预警和事件的阈值参数。
    alert_method, event_method : str or None
        分别指定预警和事件的阈值计算方法。若为 None 则使用 ``method``。
    alert_level, event_level : float or None
        分别指定预警和事件的阈值参数。若为 None 则使用 ``level``。

    Returns
    -------
    pd.DataFrame
        ``model, auc, hit_rate, fa_rate, precision, recall, f1,
        n_alerts, n_events, tp, fp, fn, tn``
    """
    # 训练数据
    train_data = _get_train_data(bundle, task)
    target = task.target[0]

    am = alert_method or method
    em = event_method or method
    al = alert_level if alert_level is not None else level
    el = event_level if event_level is not None else level

    alert_threshold = _compute_threshold(train_data, target, am, al)
    event_threshold = _compute_threshold(train_data, target, em, el)

    records: list[dict] = []
    test_predictions = getattr(vault, "test_predictions", {})

    if not test_predictions:
        raise ValueError("vault.test_predictions 为空，请确认 compare() 返回了测试集预测。")

    for model_name, td in test_predictions.items():
        y_pred = td["y_pred"].values
        y_true = td["y_true"].values

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
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(metrics).sort_values("auc", ascending=False)


# ── 阈值计算 ──────────────────────────────────────────────


def _compute_threshold(data: pd.DataFrame, target: str,
                       method: str, level: float) -> float:
    """根据方法计算阈值。

    Parameters
    ----------
    method : str
        ``"quantile"``, ``"zscore"``, 或 ``"threshold"``。
    level : float
        方法参数：
        - quantile: 分位数（0.90 = 90% 分位）
        - zscore:   标准差倍数（1.0 = 均值 + 1×std）
        - threshold: 绝对值
    """
    vals = data[target].values.astype(float)

    if method == "quantile":
        return float(np.nanquantile(vals, level))
    elif method == "zscore":
        mean = np.nanmean(vals)
        std = np.nanstd(vals) or 1.0
        return float(mean + level * std)
    elif method == "threshold":
        return float(level)
    else:
        raise ValueError(f"不支持的方法: {method}")


def _get_train_data(bundle, task):
    train = bundle.train_df.copy()
    if bundle.val_df is not None:
        train = pd.concat([train, bundle.val_df], ignore_index=True)
    return train
