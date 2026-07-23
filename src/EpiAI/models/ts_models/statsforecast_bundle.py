"""
StatsForecast models — all registered via ``register_backend`` in one shot.

Naming convention::

    sf_<algorithm_name>

Prefix ``sf_`` = StatsForecast, making the backend origin clear.
The algorithm name is descriptive snake_case.

Example::

    from EpiAI.models.registry import get

    m = get("sf_auto_arima")(season_length=12)
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
    register_backend("sf_auto_arima", AutoARIMA,
                     defaults=dict(season_length=12), exogenous=True)
    register_backend("sf_auto_ets", AutoETS,
                     defaults=dict(season_length=12, model="ZZZ"))
    register_backend("sf_auto_ces", AutoCES,
                     defaults=dict(season_length=12))
    register_backend("sf_auto_theta", AutoTheta,
                     defaults=dict(season_length=12))
    register_backend("sf_auto_tbats", AutoTBATS)

    # ── ARIMA family ────────────────────────────────────────────────
    register_backend("sf_arima", ARIMA,
                     defaults=dict(season_length=12), exogenous=True)
    register_backend("sf_auto_regressive", AutoRegressive,
                     defaults=dict(lags=12), exogenous=True)

    # ── Theta family ────────────────────────────────────────────────
    register_backend("sf_theta", Theta, defaults=dict(season_length=12))
    register_backend("sf_theta_optimized", OptimizedTheta,
                     defaults=dict(season_length=12))
    register_backend("sf_theta_dynamic", DynamicTheta,
                     defaults=dict(season_length=12))
    register_backend("sf_theta_dynamic_optimized", DynamicOptimizedTheta,
                     defaults=dict(season_length=12))

    # ── Multiple seasonalities ──────────────────────────────────────
    register_backend("sf_mstl", MSTL,
                     defaults=dict(season_length=12))
    register_backend("sf_tbats", TBATS)

    # ── GARCH / volatility ──────────────────────────────────────────
    register_backend("sf_garch", GARCH)
    register_backend("sf_arch", ARCH)

    # ── Baselines ───────────────────────────────────────────────────
    register_backend("sf_naive", Naive)
    register_backend("sf_seasonal_naive", SeasonalNaive,
                     defaults=dict(season_length=12))
    register_backend("sf_historic_average", HistoricAverage)
    register_backend("sf_random_walk_drift", RandomWalkWithDrift)
    register_backend("sf_window_average", WindowAverage,
                     defaults=dict(window_size=12))
    register_backend("sf_seasonal_window_average", SeasonalWindowAverage,
                     defaults=dict(season_length=12, window_size=12))

    # ── Exponential smoothing ───────────────────────────────────────
    register_backend("sf_simple_exp_smoothing", SimpleExponentialSmoothing,
                     defaults=dict(alpha=0.1))
    register_backend("sf_simple_exp_smoothing_optimized",
                     SimpleExponentialSmoothingOptimized)
    register_backend("sf_seasonal_exp_smoothing", SeasonalExponentialSmoothing,
                     defaults=dict(season_length=12, alpha=0.1))
    register_backend("sf_seasonal_exp_smoothing_optimized",
                     SeasonalExponentialSmoothingOptimized,
                     defaults=dict(season_length=12))
    register_backend("sf_holt", Holt,
                     defaults=dict(season_length=12))
    register_backend("sf_holt_winters", HoltWinters,
                     defaults=dict(season_length=12))

    # ── Sparse / intermittent ───────────────────────────────────────
    register_backend("sf_croston_classic", CrostonClassic)
    register_backend("sf_croston_optimized", CrostonOptimized)
    register_backend("sf_croston_sba", CrostonSBA)
    register_backend("sf_adida", ADIDA)
    register_backend("sf_imapa", IMAPA)
    register_backend("sf_tsb", TSB,
                     defaults=dict(alpha_d=0.2, alpha_p=0.2))
