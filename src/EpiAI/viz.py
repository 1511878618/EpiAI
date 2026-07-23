"""
Visualisation helpers for EpiAI — test set comparison & forecast plots.

Usage::

    from EpiAI.viz import plot_test_predictions, plot_forecast

    # After comparing models
    plot_test_predictions(bundle, vault)
    plot_test_predictions(bundle, vault, models=["sf_holt_winters", "prophet"])

    # After deployment
    hist = runtime.data_table.tail(20)
    pred = runtime.predict(horizon=12)
    plot_forecast(hist, pred, metrics=runtime.vault.metrics)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "SimHei"  # 中文标签
    matplotlib.rcParams["axes.unicode_minus"] = False
except ImportError:
    plt = None


def plot_test_predictions(
    bundle,
    vault,
    models: list[str] | None = None,
    figsize=(9, 4),
    dpi=300,
    title: str = "测试集预测对比",
):
    """绘制各模型在测试集上的预测值 vs 真实值。

    Parameters
    ----------
    bundle : PipelineBundle
        ``task.build_bundle()`` 返回的 bundle。
    vault : ModelVault
        ``task.compare()`` 返回的 vault。
    models : list of str or None
        要绘制的模型名。默认绘制所有模型。
    figsize, dpi : 图形尺寸。
    title : 标题。
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    # 取测试集时间范围
    all_time = (
        list(bundle.train_df["time"])
        + list(bundle.val_df["time"])
        + list(bundle.test_df["time"])
    )
    test_start = len(bundle.train_df) + len(bundle.val_df)
    test_time = pd.to_datetime(all_time[test_start:])

    # 真实值（逆变换后）
    y_true = bundle.get_y_series_inverse("test").ravel()

    # 待绘制的模型
    model_names = models or sorted(vault.metrics.index)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(test_time, y_true, "o-", label="真实值", color="black",
            alpha=0.8, linewidth=2, markersize=3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for name, color in zip(model_names, colors):
        df = vault.test_predictions.get(name)
        if df is None:
            continue
        yp = df["y_pred"].values
        r = vault.metrics.loc[name, "PearsonR"]
        label = f"{name} (r={r:.3f})" if not np.isnan(r) else f"{name} (r=NaN)"
        ax.plot(test_time[: len(yp)], yp, "s--", label=label,
                color=color, alpha=0.6, markersize=3)

    ax.set_xlim(test_time[0], test_time[-1])
    ax.set_ylabel("病例数")
    ax.set_xlabel("时间")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_forecast(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
    time_col: str = "time",
    target_col: str = "cases",
    top_n: int = 5,
    figsize=(12, 5),
    dpi=300,
    title: str = "未来预测",
):
    """绘制历史曲线 + 多模型未来预测。

    Parameters
    ----------
    hist_df : pd.DataFrame
        历史数据（最近 N 行，含 time_col 和 target_col）。
    pred_df : pd.DataFrame
        ``runtime.predict(horizon=N)`` 的输出。
    metrics_df : pd.DataFrame or None, optional
        ``vault.metrics``，用于在图例中显示 R²。
    time_col : str
        时间列名。
    target_col : str
        真实值列名。
    top_n : int
        最多显示的模型数。
    figsize, dpi : 图形尺寸。
    title : 标题。
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    hist_times = pd.to_datetime(hist_df[time_col])
    pred_times = pd.to_datetime(pred_df[time_col])

    # 选取模型列（排除 time 和 _risk / _lo / _hi 辅助列）
    model_cols = [
        c
        for c in pred_df.columns
        if c != time_col
        and not c.endswith("_risk")
        and not c.endswith("_lo")
        and not c.endswith("_hi")
        and not c.endswith("_quantile")
        and not c.endswith("_zscore")
    ][:top_n]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 历史曲线
    hist_values = (
        hist_df[target_col].values
        if target_col in hist_df.columns
        else hist_df[model_cols[0]].values
    )
    ax.plot(hist_times, hist_values, "o-", label="历史", color="black",
            alpha=0.8, linewidth=2, markersize=2)

    # 各模型预测
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_cols)))
    for name, color in zip(model_cols, colors):
        vals = pred_df[name].values
        r_text = ""
        if metrics_df is not None and name in metrics_df.index:
            r = metrics_df.loc[name, "PearsonR"]
            if not np.isnan(r):
                r_text = f" (r={r:.3f})"
        ax.plot(pred_times, vals, "s--", label=f"{name}{r_text}",
                color=color, alpha=0.7, markersize=4)

    ax.set_ylabel("病例数")
    ax.set_xlabel("时间")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
