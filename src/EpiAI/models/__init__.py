"""
EpiAI model zoo.

All model families are auto-imported so ``list_models()`` reflects the
full registry without additional imports::

    from EpiAI.models import get, list_models
    list_models()                          # → all 39+ models
    model = get("sf_auto_arima")(...)      # statsforecast backend
    model = get("xgb")(...)                # sklearn backend
"""
from __future__ import annotations

from .registry import get, list_models, register

# Auto-import all model families to populate the registry
from . import ts_models       # noqa: F401 — built-in TS + 33 statsforecast
from . import sklearn_models  # noqa: F401 — XGB, LGBM, RF, SVR, LinearReg

try:
    from . import torch_models  # noqa: F401 — LSTM, TCN, Transformer, …
except ImportError:
    pass  # PyTorch optional

__all__ = ["register", "get", "list_models"]

