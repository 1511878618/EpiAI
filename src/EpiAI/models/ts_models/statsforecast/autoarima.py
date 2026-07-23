"""
StatsForecast AutoARIMA forecaster — drop-in wrapper for EpiAI's TS paradigm.

Replaces the current pmdarima-based ``ARIMA`` with Nixtla's **20× faster**
C++-optimized implementation while preserving the same ``fit_sequence`` /
``predict_sequence`` / ``forecast`` interface.

Registered name: ``"AutoARIMA"`` (distinct from the original ``"ARIMA"``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register

try:
    from statsforecast.models import AutoARIMA as _SF_AutoARIMA
except ImportError:
    _SF_AutoARIMA = None  # type: ignore[assignment]


@register("AutoARIMA")
class StatsAutoARIMA(TSMixin):
    """StatsForecast AutoARIMA — univariate with optional exogenous regressors.

    Automatically selects ARIMA orders (p, d, q, P, D, Q) via AICc.
    Uses Nixtla's heavily optimised C++ backend (20× faster than pmdarima).

    Parameters
    ----------
    season_length : int, default=12
        Number of periods per season (e.g. 12 for monthly data,
        4 for quarterly, 52 for weekly, etc.).
    clip_negative : bool, default=True
        Clip negative predictions to zero (useful for case counts).
    prediction_intervals : int or None, default=None
        If set (e.g. 100), enables conformal prediction intervals
        at fit time via jackknife resampling.
    **kwargs
        Extra parameters forwarded to ``statsforecast.models.AutoARIMA``.
        See the statsforecast docs for the full list (``d``, ``D``,
        ``max_p``, ``max_q``, ``ic``, ``stepwise``, …).
    """

    # ── Capability declarations ──────────────────────────────────
    target_dim_supported = 1
    exogenous_supported = True

    def __init__(
        self,
        season_length: int = 12,
        clip_negative: bool = True,
        prediction_intervals: int | None = None,
        **kwargs,
    ) -> None:
        if _SF_AutoARIMA is None:
            raise ImportError(
                "StatsForecast is required. Install with: pip install statsforecast"
            )

        self.season_length = season_length
        self.clip_negative = clip_negative
        self.prediction_intervals = prediction_intervals
        self._model_kwargs = kwargs

        # Internal state
        self._fitted: _SF_AutoARIMA | None = None
        self._last_ds: pd.Timestamp | None = None  # set by trainer
        self._y_train: np.ndarray | None = None

    # ── Data preparation (mirrors TSMixin conventions) ───────────

    @staticmethod
    def _to_1d(arr) -> np.ndarray:
        """Coerce to 1-D float64 numpy array."""
        y = np.asarray(arr, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"Expected 1-D y, got shape {y.shape}")
        if np.isnan(y).any():
            raise ValueError("y contains NaN. Please impute or remove missing values.")
        return y

    @staticmethod
    def _to_2d(arr, T: int) -> np.ndarray | None:
        """Validate and return 2-D X, or None."""
        if arr is None:
            return None
        X = np.asarray(arr, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D X, got shape {X.shape}")
        if X.shape[0] != T:
            raise ValueError(f"X rows ({X.shape[0]}) != y length ({T})")
        if np.isnan(X).any():
            raise ValueError("X contains NaN. Please impute or remove missing values.")
        return X

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        """Clip negative values if configured."""
        if self.clip_negative:
            return np.clip(pred, 0, None)
        return pred

    # ── TSMixin interface ────────────────────────────────────────

    def fit_sequence(self, y_train, X_train=None, **kwargs):
        """Fit AutoARIMA on the full training sequence.

        Parameters
        ----------
        y_train : array-like, shape ``(T,)`` or ``(T, 1)``
            Training target values.
        X_train : array-like, shape ``(T, n_features)``, optional
            Exogenous regressors.
        """
        y = self._to_1d(y_train)
        X = self._to_2d(X_train, len(y))

        self._y_train = y.copy()

        model = _SF_AutoARIMA(
            season_length=self.season_length,
            prediction_intervals=self.prediction_intervals,
            **self._model_kwargs,
        )
        model.fit(y, X=X)
        self._fitted = model
        return self

    def predict_sequence(self, y_test, X_test=None, **kwargs):
        """Forecast the entire test period in one shot.

        Unlike the rolling-origin ``ARIMA`` forecaster, this performs a
        single ``predict(h=len(y_test))`` from the fitted model — matching
        the behaviour of ``ETS``, ``Prophet``, etc.  The speed of
        StatsForecast makes batch re-fitting cheap if rolling evaluation
        is needed later.

        Parameters
        ----------
        y_test : array-like, shape ``(n_test,)``
            Test target values (used only for length).
        X_test : array-like, shape ``(n_test, n_features)``, optional
            Future exogenous regressors.

        Returns
        -------
        np.ndarray, shape ``(n_test,)``
        """
        if self._fitted is None:
            raise RuntimeError("Model not fitted. Call fit_sequence() first.")

        n_test = len(self._to_1d(y_test))
        X = self._to_2d(X_test, n_test)
        raw = self._fitted.predict(h=n_test, X=X)
        return self._clip(np.asarray(raw["mean"], dtype=float))

    def forecast(self, n_periods: int, X_future=None):
        """Forecast truly unknown future values.

        Parameters
        ----------
        n_periods : int
            Number of steps ahead to forecast.
        X_future : array-like, shape ``(n_periods, n_features)``, optional
            Future exogenous regressors (required if model was fitted
            with ``X_train``).

        Returns
        -------
        np.ndarray, shape ``(n_periods, 1, 1)``
        """
        if self._fitted is None:
            raise RuntimeError("Model not fitted. Call fit_sequence() first.")

        X = self._to_2d(X_future, n_periods)
        raw = self._fitted.predict(h=n_periods, X=X)
        pred = self._clip(np.asarray(raw["mean"], dtype=float))
        return pred.reshape(-1, 1, 1)

    def __repr__(self) -> str:
        return (
            f"StatsAutoARIMA(season_length={self.season_length}, "
            f"clip={self.clip_negative})"
        )
