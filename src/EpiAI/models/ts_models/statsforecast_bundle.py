"""
StatsForecast models — all registered via ``register_backend`` in one shot.

Each model is registered with a unique name.  The convention is::

    stats/{original_name}

Example::

    from EpiAI.models.registry import get

    m = get("stats/autoarima")(season_length=12)
    m.fit_sequence(y_train)
    m.forecast(12)
"""
from __future__ import annotations

from ..adapter import register_backend

try:
    from statsforecast.models import (
        # ── Auto / search ──────────────────────────────────────────
        AutoARIMA,
        AutoETS,
        AutoCES,
        AutoTheta,
        AutoTBATS,
        # ── ARIMA family ────────────────────────────────────────────
        ARIMA,
        AutoRegressive,
        # ── Theta family ────────────────────────────────────────────
        Theta,
        OptimizedTheta,
        DynamicTheta,
        DynamicOptimizedTheta,
        # ── Multiple seasonalities ──────────────────────────────────
        MSTL,
        TBATS,
        # ── GARCH / volatility ──────────────────────────────────────
        GARCH,
        ARCH,
        # ── Baselines ───────────────────────────────────────────────
        Naive,
        SeasonalNaive,
        HistoricAverage,
        RandomWalkWithDrift,
        WindowAverage,
        SeasonalWindowAverage,
        # ── Exponential smoothing ────────────────────────────────────
        SimpleExponentialSmoothing,
        SimpleExponentialSmoothingOptimized,
        SeasonalExponentialSmoothing,
        SeasonalExponentialSmoothingOptimized,
        Holt,
        HoltWinters,
        # ── Sparse / intermittent ──────────────────────────────────
        CrostonClassic,
        CrostonOptimized,
        CrostonSBA,
        ADIDA,
        IMAPA,
        TSB,
    )
    _HAS_STATSFORECAST = True

except ImportError:
    _HAS_STATSFORECAST = False


if _HAS_STATSFORECAST:
    # ── Auto / search models ────────────────────────────────────────
    register_backend("stats/autoarima", AutoARIMA,
                     defaults=dict(season_length=12), exogenous=True)
    register_backend("stats/autoets", AutoETS,
                     defaults=dict(season_length=12, model="ZZZ"))
    register_backend("stats/autoces", AutoCES,
                     defaults=dict(season_length=12))
    register_backend("stats/autotheta", AutoTheta,
                     defaults=dict(season_length=12))
    register_backend("stats/autotbats", AutoTBATS)

    # ── ARIMA family ────────────────────────────────────────────────
    register_backend("stats/arima", ARIMA,
                     defaults=dict(season_length=12), exogenous=True)
    register_backend("stats/autoregressive", AutoRegressive,
                     defaults=dict(lags=12), exogenous=True)

    # ── Theta family ────────────────────────────────────────────────
    register_backend("stats/theta", Theta, defaults=dict(season_length=12))
    register_backend("stats/opttheta", OptimizedTheta,
                     defaults=dict(season_length=12))
    register_backend("stats/dyntheta", DynamicTheta,
                     defaults=dict(season_length=12))
    register_backend("stats/dynopttheta", DynamicOptimizedTheta,
                     defaults=dict(season_length=12))

    # ── Multiple seasonalities ──────────────────────────────────────
    register_backend("stats/mstl", MSTL,
                     defaults=dict(season_length=12))
    register_backend("stats/tbats", TBATS)

    # ── GARCH / volatility ──────────────────────────────────────────
    register_backend("stats/garch", GARCH)
    register_backend("stats/arch", ARCH)

    # ── Baselines ───────────────────────────────────────────────────
    register_backend("stats/naive", Naive)
    register_backend("stats/snaive", SeasonalNaive,
                     defaults=dict(season_length=12))
    register_backend("stats/histavg", HistoricAverage)
    register_backend("stats/rwdrift", RandomWalkWithDrift)
    register_backend("stats/winavg", WindowAverage,
                     defaults=dict(window_size=12))
    register_backend("stats/swinavg", SeasonalWindowAverage,
                     defaults=dict(season_length=12, window_size=12))

    # ── Exponential smoothing ───────────────────────────────────────
    register_backend("stats/ses", SimpleExponentialSmoothing,
                     defaults=dict(alpha=0.1))
    register_backend("stats/sesopt", SimpleExponentialSmoothingOptimized)
    register_backend("stats/seasonal_ses", SeasonalExponentialSmoothing,
                     defaults=dict(season_length=12, alpha=0.1))
    register_backend("stats/seasonal_sesopt",
                     SeasonalExponentialSmoothingOptimized,
                     defaults=dict(season_length=12))
    register_backend("stats/holt", Holt,
                     defaults=dict(season_length=12))
    register_backend("stats/hw", HoltWinters,
                     defaults=dict(season_length=12))

    # ── Sparse / intermittent ───────────────────────────────────────
    register_backend("stats/croston", CrostonClassic)
    register_backend("stats/crostonopt", CrostonOptimized)
    register_backend("stats/crostonsba", CrostonSBA)
    register_backend("stats/adida", ADIDA)
    register_backend("stats/imapa", IMAPA)
    register_backend("stats/tsb", TSB,
                     defaults=dict(alpha_d=0.2, alpha_p=0.2))
