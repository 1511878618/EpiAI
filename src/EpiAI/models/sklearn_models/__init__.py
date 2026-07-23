"""Sklearn-compatible forecasting models.

The actual model backends (lightgbm, xgboost) are imported
lazily inside each class to allow ``@register`` to work without them.
"""
from __future__ import annotations

from .RF import RandomForestForecaster
from .glm import LinearRegForecaster
from .lgbm import LGBMSingleForecaster
from .svm import SVRForecaster
from .xgb import XGBSingleForecaster

__all__ = [
    "LGBMSingleForecaster",
    "XGBSingleForecaster",
    "RandomForestForecaster",
    "SVRForecaster",
    "LinearRegForecaster",
]
