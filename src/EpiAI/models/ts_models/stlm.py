"""
STLM-ARIMA forecaster.

STL decomposition + ARIMA on the seasonally-adjusted component.
Mirrors R's ``forecast::stlm()`` with ``s.window = "periodic"``.

Pipeline:
  1. STL decompose y = trend + seasonal + residual
  2. Fit ARIMA on (trend + residual) = seasonally-adjusted series
  3. Forecast: ARIMA forecast + seasonal (repeated last seasonal cycle)

Requires ``pmdarima`` and ``statsmodels``.
"""
from __future__ import annotations

import numpy as np

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register

try:
    import pmdarima as pm
except ImportError:
    pm = None  # type: ignore[assignment]

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None  # type: ignore[assignment]


@register("STLM")
class STLMARIMAForecaster(TSMixin):
    """STL decomposition + ARIMA on seasonally-adjusted component.

    Parameters
    ----------
    seasonal_periods : int, default=12
        Seasonal period (e.g. 12 for monthly data).
    seasonal : str, default="periodic"
        STL seasonal window. ``"periodic"`` uses the same window
        each year. An integer forces a rolling seasonal window.
    clip_negative : bool, default=True
        Clip negative predictions to zero.
    auto_arima_kwargs : dict, optional
        Extra kwargs for ``pmdarima.auto_arima``.
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        seasonal: str = "periodic",
        clip_negative: bool = True,
        **auto_arima_kwargs,
    ) -> None:
        if pm is None:
            raise ImportError("STLM-ARIMA requires pmdarima. pip install pmdarima")
        if STL is None:
            raise ImportError("STLM-ARIMA requires statsmodels. pip install statsmodels")

        self.seasonal_periods = seasonal_periods
        self.seasonal = seasonal
        self.clip_negative = clip_negative
        self.auto_arima_kwargs = auto_arima_kwargs

        self._seasonal_component: np.ndarray | None = None
        self._arima_model: pm.arima.ARIMA | None = None
        self._y_train: np.ndarray | None = None

    def fit_sequence(self, y_train, X_train=None, **kwargs):
        """Fit STLM-ARIMA on training sequence.

        Parameters
        ----------
        y_train : array-like, shape (n,)
        X_train : ignored
        """
        y_train = np.asarray(y_train, dtype=float)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train[:, 0]
        if y_train.ndim != 1:
            raise ValueError(f"y_train must be 1D, got {y_train.shape}")

        self._y_train = y_train.copy()
        n = len(y_train)

        # STL decomposition — "periodic" → use a large odd seasonal window
        # to approximate periodic behavior (all years contribute equally)
        if self.seasonal == "periodic":
            seasonal_win = max(self.seasonal_periods * 2 + 1, 7)
            # Ensure odd
            if seasonal_win % 2 == 0:
                seasonal_win += 1
        else:
            seasonal_win = int(self.seasonal)
        stl = STL(
            y_train,
            period=self.seasonal_periods,
            seasonal=seasonal_win,
        )
        res = stl.fit()

        seasonal = res.seasonal
        seasonally_adjusted = res.trend + res.resid

        # Fit ARIMA on seasonally-adjusted series
        self._seasonal_component = seasonal[-self.seasonal_periods:].copy()

        kwargs = dict(
            seasonal=False,
            stepwise=True,
            trace=False,
            suppress_warnings=True,
        )
        kwargs.update(self.auto_arima_kwargs)

        self._arima_model = pm.auto_arima(seasonally_adjusted, **kwargs)
        return self

    def _predict_with_seasonal(self, n: int) -> np.ndarray:
        """Generate predictions and add back seasonal component."""
        if self._arima_model is None or self._seasonal_component is None:
            raise RuntimeError("Model not fitted yet.")

        # Forecast seasonally-adjusted series
        fcst, conf_int = self._arima_model.predict(
            n_periods=n, return_conf_int=True,
        )
        fcst = np.asarray(fcst, dtype=float)

        # Repeat seasonal cycle
        seasonal_future = np.tile(
            self._seasonal_component,
            int(np.ceil(n / self.seasonal_periods)),
        )[:n]

        pred = fcst + seasonal_future
        if self.clip_negative:
            pred = np.clip(pred, 0, None)
        return pred

    def predict_sequence(self, y_test, X_test=None, update_state=True, **kwargs):
        """Predict on test sequence.

        Parameters
        ----------
        y_test : array-like, shape (n_test,)
        X_test : ignored
        update_state : bool, optional
            If True, append test data and refit.

        Returns
        -------
        np.ndarray, shape (n_test,)
        """
        n_test = len(y_test)
        pred = self._predict_with_seasonal(n_test)

        if update_state:
            self._y_train = np.concatenate([self._y_train, np.asarray(y_test, dtype=float)])
            self.fit_sequence(self._y_train)

        return pred

    def forecast(self, n_periods: int, X_future=None):
        """Forecast future values.

        Parameters
        ----------
        n_periods : int
        X_future : ignored

        Returns
        -------
        np.ndarray, shape (n_periods, 1, 1)
        """
        pred = self._predict_with_seasonal(n_periods)
        return pred.reshape(-1, 1, 1)

    def __repr__(self) -> str:
        return (
            f"STLMARIMAForecaster(period={self.seasonal_periods})"
        )
