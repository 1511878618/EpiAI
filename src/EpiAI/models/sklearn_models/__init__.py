"""Sklearn-compatible forecasting models.

The actual model backends (lightgbm, xgboost, tabpfn) are imported
lazily inside each class to allow ``@register`` to work without them.
"""
from __future__ import annotations

from .RF import RandomForestForecaster
from .glm import LinearRegForecaster
from .lgbm import LGBMSingleForecaster
from .svm import SVRForecaster
from .tabpfn import TabPFNMultiForecaster
from .xgb import XGBSingleForecaster

__all__ = [
    "LGBMSingleForecaster",
    "XGBSingleForecaster",
    "TabPFNMultiForecaster",
    "RandomForestForecaster",
    "SVRForecaster",
    "LinearRegForecaster",
]
