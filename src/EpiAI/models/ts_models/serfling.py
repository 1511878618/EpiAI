"""
Serfling regression forecaster.

Serfling 回归是经典的传染病基线模型：
  y = trend + Fourier seasonal terms + noise

Fits on the full training sequence, forecasts forward.
No exogenous variables supported.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register


@register("Serfling")
class SerflingForecaster(TSMixin):
    """Serfling regression: trend + Fourier seasonal terms.

    Parameters
    ----------
    seasonal_periods : int, default=12
        Seasonal period (e.g. 12 for monthly data).
    fourier_order : int, default=2
        Number of Fourier sin/cos pairs for seasonal pattern.
        Higher values capture more complex seasonality.
    trend_order : int, default=1
        Polynomial order of the trend component.
        1 = linear trend, 2 = quadratic, etc.
    clip_negative : bool, default=True
        Whether to clip negative predictions to zero.
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        fourier_order: int = 2,
        trend_order: int = 1,
        clip_negative: bool = True,
    ) -> None:
        self.seasonal_periods = seasonal_periods
        self.fourier_order = fourier_order
        self.trend_order = trend_order
        self.clip_negative = clip_negative
        self.model_: LinearRegression | None = None
        self._y_train: np.ndarray | None = None

    def _build_features(self, t: np.ndarray) -> np.ndarray:
        """Build design matrix: trend + Fourier terms."""
        features = []
        # Trend (polynomial)
        for p in range(1, self.trend_order + 1):
            features.append(t ** p)
        # Fourier seasonal terms
        for k in range(1, self.fourier_order + 1):
            features.append(np.sin(2 * np.pi * k * t / self.seasonal_periods))
            features.append(np.cos(2 * np.pi * k * t / self.seasonal_periods))
        return np.column_stack(features)

    def fit_sequence(self, y_train, X_train=None, **kwargs):
        """Fit Serfling regression on training sequence.

        Parameters
        ----------
        y_train : array-like, shape (n,)
            Training target values.
        X_train : ignored
            Serfling does not support exogenous variables.
        """
        y_train = np.asarray(y_train, dtype=float)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train[:, 0]
        if y_train.ndim != 1:
            raise ValueError(f"y_train must be 1D, got {y_train.shape}")

        self._y_train = y_train.copy()
        n = len(y_train)
        t = np.arange(n)
        X = self._build_features(t)

        self.model_ = LinearRegression()
        self.model_.fit(X, y_train)
        return self

    def predict_sequence(self, y_test, X_test=None, update_state=True, **kwargs):
        """Predict on test sequence using the fitted model.

        Parameters
        ----------
        y_test : array-like, shape (n_test,)
            Test target values (used only for length).
        X_test : ignored
        update_state : bool, optional
            If True, append y_test to training history for subsequent
            forecasts.

        Returns
        -------
        np.ndarray, shape (n_test,)
        """
        n_train = len(self._y_train)
        n_test = len(y_test)
        t = np.arange(n_train, n_train + n_test)
        X = self._build_features(t)

        pred = self.model_.predict(X)
        if self.clip_negative:
            pred = np.clip(pred, 0, None)

        if update_state:
            self._y_train = np.concatenate([self._y_train, y_test])

        return pred

    def forecast(self, n_periods: int, X_future=None):
        """Forecast future values.

        Parameters
        ----------
        n_periods : int
            Number of steps to forecast.
        X_future : ignored

        Returns
        -------
        np.ndarray, shape (n_periods, 1, 1)
        """
        n_train = len(self._y_train)
        t = np.arange(n_train, n_train + n_periods)
        X = self._build_features(t)

        pred = self.model_.predict(X)
        if self.clip_negative:
            pred = np.clip(pred, 0, None)

        return pred.reshape(-1, 1, 1)

    def __repr__(self) -> str:
        return (
            f"SerflingForecaster("
            f"order={self.trend_order}, "
            f"fourier_K={self.fourier_order}, "
            f"period={self.seasonal_periods})"
        )
