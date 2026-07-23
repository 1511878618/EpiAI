"""Time-series statistical forecasting models (ARIMA, ETS, Serfling, …).

Includes built-in models + statsforecast backends.
"""
from __future__ import annotations

from .ARIMA import AutoARIMAXRollingForecaster
from .ETS import ETSForecaster
from .serfling import SerflingForecaster
from .prophet import ProphetForecaster
from .stlm import STLMARIMAForecaster
from .bsts import BSTSForecaster

# StatsForecast backends (33 models) — registered when this module loads
# noinspection PyUnresolvedReferences
from . import statsforecast_bundle  # noqa: F401

__all__ = [
    "AutoARIMAXRollingForecaster",
    "ETSForecaster",
    "SerflingForecaster",
    "ProphetForecaster",
    "STLMARIMAForecaster",
    "BSTSForecaster",
]
