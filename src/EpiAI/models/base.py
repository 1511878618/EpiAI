"""
Base classes for all EpiAI forecasters.

Defines three model families via mixins:

* ``TorchMixin``   — PyTorch ``nn.Module``, gradient descent training
* ``SklearnMixin`` — sklearn-compatible, ``.fit(x, y)`` / ``.predict(x)``
* ``TSMixin``      — Time-series statistical models (ARIMA, ETS, …)

Usage::

    @register("LSTM")
    class LSTMForecaster(nn.Module, TorchMixin):
        ...

    @register("XGB")
    class XGBSingleForecaster(SklearnMixin):
        ...

    @register("ARIMA")
    class AutoARIMAXRollingForecaster(TSMixin):
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import numpy as np


# =====================================================================
# BaseForecaster
# =====================================================================

class BaseForecaster(ABC):
    """Every EpiAI model inherits from this ABC via one of the mixins.

    The ``paradigm`` class-method tells ``EpiAITrainer`` which training
    path to follow.
    """

    @classmethod
    @abstractmethod
    def paradigm(cls) -> Literal["torch", "sklearn", "ts"]:
        ...

    # ── Shared: Torch & Sklearn (windowed data) ──────────────

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fit(). "
            f"Use fit_sequence() for TS models."
        )

    def predict(self, x) -> np.ndarray:
        raise NotImplementedError

    # ── TS-specific (raw sequence) ────────────────────────────

    def fit_sequence(self, y_train, X_train=None):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fit_sequence(). "
            f"Use fit() for window-based models."
        )

    def predict_sequence(self, y_test, X_test=None,
                         update_state=True) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, n_periods: int, X_future=None) -> np.ndarray:
        raise NotImplementedError

    # ── Serialisation ─────────────────────────────────────────

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseForecaster":
        raise NotImplementedError


# =====================================================================
# Mixins
# =====================================================================

class TorchMixin(BaseForecaster):
    """Mixin for PyTorch models.

    Provides a default ``predict()`` that runs ``forward()`` and
    returns a numpy array.  Subclasses may override for custom logic.
    """

    @classmethod
    def paradigm(cls) -> Literal["torch"]:
        return "torch"

    def predict(self, x) -> np.ndarray:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for torch models.")

        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            x_t = x_t.to(next(self.parameters()).device)
            pred = self.forward(x_t)
            return pred.cpu().numpy()


class SklearnMixin(BaseForecaster):
    """Mixin for sklearn-compatible models.

    Provides helpers for flattening 3-D windows to 2-D arrays.
    """

    @classmethod
    def paradigm(cls) -> Literal["sklearn"]:
        return "sklearn"

    @staticmethod
    def _flatten_x(x: np.ndarray) -> np.ndarray:
        """(N, lookback, n_features) → (N, lookback * n_features)"""
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _prepare_y(y: np.ndarray) -> np.ndarray:
        """(N, horizon, target_dim) → (N, horizon * target_dim).

        For single-target (target_dim=1) with horizon=1, returns ``(N,)``.
        """
        if y.ndim == 3 and y.shape[1] == 1 and y.shape[2] == 1:
            return y[:, 0, 0]
        return y.reshape(y.shape[0], -1)

    @staticmethod
    def _reshape_pred(pred: np.ndarray, horizon=1, target_dim=1) -> np.ndarray:
        """(N, ...) → (N, horizon, target_dim)"""
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred.reshape(pred.shape[0], horizon, target_dim)


class TSMixin(BaseForecaster):
    """Mixin for time-series statistical models."""

    @classmethod
    def paradigm(cls) -> Literal["ts"]:
        return "ts"


__all__ = [
    "BaseForecaster",
    "TorchMixin",
    "SklearnMixin",
    "TSMixin",
]
