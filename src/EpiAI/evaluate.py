"""
预警评估模块：滚动回测检验预警准确率。

核心思路：
  对多个历史时间点做"回到过去"模拟：
  1. 用截止到该点的数据训练模型
  2. 预测未来 H 个月 + 分配风险等级
  3. 等"真实数据"出现后，比较预警是否命中

使用方法::

    from EpiAI.evaluate import warning_report

    report = warning_report(
        data=df, target="cases", time_col="time", freq="MS",
        models=["sf_auto_ets", "sf_holt_winters"],
        horizon=12, n_windows=8,
        risk_method="quantile",
        event_threshold="p95",         # 事件定义：超过历史 95% 分位
        cache_dir="./eval_cache",
    )
    print(report)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def warning_report(
    data: pd.DataFrame,
    models: list[str],
    target: str = "cases",
    time_col: str = "time",
    freq: str = "MS",
    horizon: int = 12,
    n_windows: int = 8,
    risk_method: str = "quantile",
    event_threshold: float | str = 0.95,
    warning_level: int = 2,
    cache_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """滚动回测预警准确率。

    Parameters
    ----------
    data : pd.DataFrame
        完整历史数据（含 ``time_col`` 和 ``target`` 列）。
    models : list of str
        要评估的模型注册名列表。
    target, time_col, freq : str
        数据列配置，同 ``ForecastTask``。
    horizon : int
        预测步数。
    n_windows : int
        滚动窗口数。越大回测点越多，但训练次数也越多。
    risk_method : str
        风险评分方法，透传给 ``RiskScorer``。
    event_threshold : float or str
        真实事件的定义阈值：

        - ``0.95``（默认）：实际值 > 历史同期 95% 分位 → 事件
        - ``500``（数值）：实际值 > 500 → 事件
        - ``"2x_median"``：实际值 > 历史中位数 × 2
    warning_level : int
        预警触发等级，默认 ``2``（风险 ≥ 2 视为预警）。
    cache_dir : str or None
        ``ForecastTask.compare()`` 缓存目录，避免重复训练。
    verbose : bool
        是否打印进度。

    Returns
    -------
    pd.DataFrame
        每模型 × 每 horizon 的预警评估指标：

        ``model, horizon, hit_rate, fa_rate, precision, recall, f1,
        n_alerts, n_events, tp, fp, fn, tn``
    """
    from EpiAI import ForecastTask
    from EpiAI.dataset import TimeSplit
    from EpiAI.risk import RiskScorer

    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)

    # ── 确定滚动窗口的切割点 ─────────────────────────────────
    # 每个窗口：末尾留 horizon 个月做"真实未来"
    max_train_start = n - horizon - 1
    step = max(1, max_train_start // n_windows) if n_windows > 1 else max_train_start
    if step == 0:
        raise ValueError(f"数据量({n})不足以做 {n_windows} 个回测窗口 (horizon={horizon})")

    train_sizes = list(range(step, n - horizon, step))[:n_windows]
    if verbose:
        print(f"预警回测: {len(train_sizes)} 个窗口, "
              f"horizon={horizon}, risk={risk_method}, "
              f"event_threshold={event_threshold}")

    # ── 滚动回测 ─────────────────────────────────────────────
    records: list[dict] = []

    for wi, train_size in enumerate(train_sizes):
        train_data = df.iloc[:train_size]
        test_period = df.iloc[train_size:train_size + horizon]
        if len(test_period) < horizon:
            continue

        if verbose:
            t_start = test_period[time_col].iloc[0].date()
            t_end = test_period[time_col].iloc[-1].date()
            print(f"\n  窗口 {wi + 1}/{len(train_sizes)}: "
                  f"训练 {train_data[time_col].iloc[0].date()} → "
                  f"{train_data[time_col].iloc[-1].date()}  "
                  f"测试 {t_start} → {t_end}", end="", flush=True)

        # 事件阈值（基于训练数据计算）
        actual_threshold = _resolve_threshold(train_data, target, time_col,
                                              event_threshold)

        # 风险评分器（用训练数据拟合基线）
        scorer = RiskScorer(method=risk_method, time_col=time_col,
                            value_col=target, same_period=True)
        scorer.fit(train_data)

        # ── 训练 + 预测 ──
        try:
            # 用 train_data 做 0.95/0.025 切分
            n_tr = len(train_data)
            split = TimeSplit(
                train_ratio=1.0 - 2 * horizon / n_tr,
                val_ratio=horizon / n_tr,
            )
            task = ForecastTask(
                data=train_data, target=target, time_col=time_col,
                freq=freq, horizon=horizon, split=split,
            )
            vault = task.compare(models, cache_dir=str(cache_dir) if cache_dir else None)

            best = vault.best("R2")
            runtime = task.deploy(best, vault, risk=None)
        except Exception as e:
            if verbose:
                print(f"   × 训练失败: {e}")
            continue

        # ── 预测 ──
        try:
            pred = runtime.predict(horizon=horizon)
        except Exception as e:
            if verbose:
                print(f"  × 预测失败: {e}")
            continue

        # ── 逐 horizon 比对 ──
        for h in range(horizon):
            pred_time = pred[time_col].iloc[h]
            # 找到对应的真实值
            match = test_period[test_period[time_col] == pred_time]
            if len(match) == 0:
                continue
            actual_val = match[target].values[0]

            for model_name in models:
                if model_name not in pred.columns:
                    continue
                pred_val = pred[model_name].values[h]
                risk_level = scorer.score(float(pred_val),
                                          pd.Timestamp(pred_time))
                is_event = bool(actual_val > actual_threshold)

                records.append({
                    "window": wi,
                    "model": model_name,
                    "horizon": h + 1,
                    "pred_time": pred_time,
                    "pred_value": float(pred_val),
                    "actual_value": float(actual_val),
                    "risk_level": int(risk_level),
                    "is_warning": risk_level >= warning_level,
                    "is_event": is_event,
                })

        if verbose:
            print("  ✓", end="", flush=True)

    if not records:
        raise RuntimeError("没有任何回测记录生成，请检查数据量和参数。")

    result_df = pd.DataFrame(records)

    # ── 计算评估指标 ─────────────────────────────────────────
    metrics = []
    for (model, horizon_grp), grp in result_df.groupby(["model", "horizon"]):
        warned = grp["is_warning"]
        event = grp["is_event"]
        tp = int((warned & event).sum())
        fp = int((warned & ~event).sum())
        fn = int((~warned & event).sum())
        tn = int((~warned & ~event).sum())

        def _safe_div(a, b):
            return a / b if b > 0 else np.nan

        hit_rate = _safe_div(tp, tp + fn)
        fa_rate = _safe_div(fp, fp + tn)
        precision = _safe_div(tp, tp + fp)
        recall = hit_rate
        f1 = _safe_div(2 * precision * recall, precision + recall)

        metrics.append({
            "model": model, "horizon": horizon_grp,
            "hit_rate": hit_rate, "fa_rate": fa_rate,
            "precision": precision, "recall": recall, "f1": f1,
            "n_alerts": int(warned.sum()),
            "n_events": int(event.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(metrics).sort_values(["model", "horizon"])


# ── 阈值辅助 ────────────────────────────────────────────────

def _resolve_threshold(
    train_data: pd.DataFrame, target: str, time_col: str,
    threshold: float | str,
) -> float:
    """将用户定义的阈值解析为具体数值。"""
    if isinstance(threshold, (int, float)):
        if 0 < threshold < 1:
            # 分位数
            return float(train_data[target].quantile(threshold))
        return float(threshold)

    if isinstance(threshold, str):
        if threshold.startswith("p"):
            q = float(threshold[1:]) / 100
            return float(train_data[target].quantile(q))
        if threshold == "2x_median":
            return float(train_data[target].median() * 2)

    raise ValueError(f"无法解析阈值: {threshold}")


def plot_warning_report(report: pd.DataFrame, models: Optional[list[str]] = None,
                        figsize=(10, 5), dpi=200):
    """绘制预警评估报告。

    横轴 = horizon，纵轴 = 指标值，每条线 = 一个模型。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required.")

    model_names = models or report["model"].unique()
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    colors = plt.cm.tab10.colors

    for mi, metric in enumerate(["hit_rate", "fa_rate", "f1"]):
        ax = axes[mi]
        for i, name in enumerate(model_names):
            sub = report[(report["model"] == name)]
            ax.plot(sub["horizon"], sub[metric], "o-", color=colors[i % len(colors)],
                    label=name, alpha=0.7, markersize=3)
        ax.set_title({"hit_rate": "Hit Rate (Recall)", "fa_rate": "False Alarm Rate",
                       "f1": "F1 Score"}[metric])
        ax.set_xlabel("Horizon (months ahead)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        if mi == 0:
            ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.show()
    return fig, axes
