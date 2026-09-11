"""
预警评估模块 — 多方法二分类预警评估。

支持多种方法定义阈值，然后做纯二分类：
  - 预测值超过阈值 → 预警
  - 实际值超过阈值 → 事件

方法:
  - "zscore":    阈值 = mean + level × std
  - "quantile":  阈值 = level 分位数（如 0.90）
  - "threshold": 阈值 = level（绝对值）

使用方法::（已迁移，直接走 ModelVault）

    from EpiAI import ForecastTask

    task = ForecastTask(data=df, target="cases", freq="MS", horizon=12)
    vault = task.compare(["sf_auto_ets", "sf_holt_winters"])

    # 校准式评估：Youden 最优 cutoff（固化后 predict 即按模型专属阈值报警）
    report = vault.evaluate_risk(risk=["quantile@0.90", "zscore@1.0"],
                                 persist=True)
    print(report[["strategy", "model", "auc", "cutoff", "youden", "f1"]])
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def warning_report(
    vault,
    bundle=None,
    task=None,
    method: str = "quantile",
    level: float = 0.90,
    alert_method: str | None = None,
    event_method: str | None = None,
    alert_level: float | None = None,
    event_level: float | None = None,
    persist: bool = False,
) -> pd.DataFrame:
    """利用 compare() 测试集预测做校准式预警评估（Youden cutoff）。

    已迁移为 ``ModelVault.evaluate_risk()``：请直接调用
    ``vault.evaluate_risk(risk=\"quantile@0.90\", persist=True)``。
    本函数保留旧签名仅为兼容转发。

    Parameters
    ----------
    vault : ModelVault
        ``task.compare()`` 返回值。
    method : str, default "quantile"
        事件方法（quantile / zscore / threshold）。
    level : float, default 0.90
        事件级别。
    event_method, event_level : str/float or None
        可选：显式覆盖事件定义（默认沿用 ``method``/``level``）。
    persist : bool, default False
        是否固化 cutoff 到 ``vault.risk_cutoffs``。
    alert_method, alert_level : 已废弃
        校准语义下不再存在独立的"预警统计阈值"——报警阈值由每个模型
        Youden 校准得到。传入非 None 会抛出 ValueError。

    Returns
    -------
    pd.DataFrame
        ``strategy, model, auc, cutoff, youden, sens, spec, hit_rate,
        fa_rate, precision, recall, f1, n_alerts, n_events, tp, fp, fn, tn``
    """
    import warnings
    warnings.warn(
        "evaluate.warning_report() 已迁移为 ModelVault.evaluate_risk()："
        "评估档案（test_predictions/训练阈值基准）由 vault 自身维护，"
        "无需再传 vault/bundle/task。请直接调用 vault.evaluate_risk(...)。",
        DeprecationWarning, stacklevel=2,
    )
    if alert_method is not None or alert_level is not None:
        raise ValueError(
            "alert_method/alert_level 已废弃：校准语义下报警阈值由每个模型 "
            "Youden 校准得到（evaluate_risk 返回/固化 cutoff），"
            "不再存在共用的统计预警阈值。请移除这两个参数。")
    return vault.evaluate_risk(
        method=method, level=level,
        event_method=event_method, event_level=event_level,
        persist=persist,
    )

