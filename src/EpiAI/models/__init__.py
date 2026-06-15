"""
EpiAI model zoo.

Model families
--------------
* ``torch_models`` — PyTorch deep-learning forecasters (CNN, LSTM, …)
* ``sklearn_models`` — sklearn-compatible forecasters (XGB, LGBM, …)
* ``ts_models`` — Time-series statistical models (ARIMA, ETS, …)

Quick lookup (requires specific import)::

    from EpiAI.models.registry import get, list_models

    Cls = get("LSTM")
    model = Cls(input_dim=8, lookback=12, horizon=3, target_dim=1)
"""

from __future__ import annotations

from .registry import get, list_models, register

__all__ = ["register", "get", "list_models"]
