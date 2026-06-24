"""
Prophet forecaster wrapper.

Wraps Facebook Prophet (``prophet`` package) as an EpiAI TS-paradigm model.

Requires::

    pip install prophet

Or in conda::

    conda install -c conda-forge prophet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register

try:
    from prophet import Prophet
except ImportError:
    Prophet = None  # type: ignore[assignment]


@register("Prophet")
class ProphetForecaster(TSMixin):
    """Facebook Prophet forecaster.

    Prophet is a decomposable time series model with trend,
    seasonality, and holiday effects.

    Parameters
    ----------
    growth : str, default="linear"
        Trend type: ``"linear"``, ``"logistic"``, or ``"flat"``.
    seasonality_mode : str, default="additive"
        ``"additive"`` or ``"multiplicative"``.
    yearly_seasonality : bool or int, default=True
        Yearly seasonality. ``True`` = auto, ``int`` = Fourier order.
    weekly_seasonality : bool or int, default=False
        Weekly seasonality (usually False for monthly data).
    daily_seasonality : bool or int, default=False
        Daily seasonality.
    changepoint_prior_scale : float, default=0.05
        Flexibility of the trend.
    seasonality_prior_scale : float, default=10.0
        Strength of the seasonality.
    clip_negative : bool, default=True
        Clip negative predictions to zero.
    prophet_kwargs : dict, optional
        Additional kwargs passed to ``prophet.Prophet()``.
    """

    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        yearly_seasonality: bool | int = True,
        weekly_seasonality: bool | int = False,
        daily_seasonality: bool | int = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        clip_negative: bool = True,
        **prophet_kwargs,
    ) -> None:
        if Prophet is None:
            raise ImportError(
                "Prophet is required. Install with: pip install prophet"
            )

        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.clip_negative = clip_negative
        self.prophet_kwargs = prophet_kwargs

        self._model: Prophet | None = None
        self._last_ds: pd.Timestamp | None = None

    def fit_sequence(self, y_train, X_train=None, dates=None, **kwargs):
        """Fit Prophet on training sequence.

        Parameters
        ----------
        y_train : array-like, shape (n,)
            Training target values.
        X_train : ignored
            Prophet does not use X_train directly; use
            ``prophet_kwargs`` to add extra regressors instead.
        dates : array-like, optional
            Real date stamps corresponding to y_train. If None,
            synthetic dates are generated.
        """
        y_train = np.asarray(y_train, dtype=float)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train[:, 0]
        y_train = np.asarray(y_train).ravel()

        n = len(y_train)
        if n > 0 and np.nanmax(y_train) == np.nanmin(y_train):
            y_train = y_train + np.random.default_rng(42).normal(0, 1e-6, n)

        # Use real dates when available
        if dates is not None and len(dates) == n:
            ds = pd.DatetimeIndex(pd.to_datetime(dates))
        else:
            from pandas.tseries.offsets import MonthBegin
            today_norm = pd.Timestamp.today().normalize()
            last_ms = today_norm + MonthBegin(-1)
            _dates = [last_ms + MonthBegin(-i) for i in range(n)]
            ds = pd.DatetimeIndex(reversed(_dates))

        df = pd.DataFrame({"ds": ds, "y": y_train})

        self._model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            **self.prophet_kwargs,
        )
        self._model.fit(df)
        self._last_ds = ds[-1]
        return self

    def _make_future(self, n_periods: int) -> pd.DataFrame:
        """Create future DataFrame for Prophet prediction."""
        if self._model is None or self._last_ds is None:
            raise RuntimeError("Model not fitted yet.")
        future = self._model.make_future_dataframe(
            periods=n_periods, freq="MS", include_history=False,
        )
        return future

    def predict_sequence(self, y_test, X_test=None, update_state=True, dates=None, **kwargs):
        """Predict on test sequence.

        Parameters
        ----------
        y_test : array-like, shape (n_test,)
            Test values (used for length).
        X_test : ignored
        update_state : bool, optional
            If True, append test values and refit (online update).
        dates : array-like, optional
            Real test dates. When provided, used to compute the exact
            number of future periods needed, compensating for any gap
            between training end and test start (e.g. validation set).

        Returns
        -------
        np.ndarray, shape (n_test,)
        """
        n_test = len(y_test)
        if dates is not None and len(dates) > 0 and self._last_ds is not None:
            # Use test dates to compute total periods from train end to test end
            test_end = pd.to_datetime(dates[-1])
            n_total = ((test_end.year - self._last_ds.year) * 12 +
                       (test_end.month - self._last_ds.month))
            future = self._model.make_future_dataframe(
                periods=n_total, freq="MS", include_history=False,
            )
            fcst = self._model.predict(future)
            # Select only predictions matching test date range
            test_start = pd.to_datetime(dates[0])
            mask = fcst["ds"].between(test_start, test_end)
            pred = fcst.loc[mask, "yhat"].values[:n_test]
        else:
            # Fallback: assume consecutive dates (no validation gap)
            future = self._make_future(n_test)
            fcst = self._model.predict(future)
            pred = fcst["yhat"].values
        if self.clip_negative:
            pred = np.clip(pred, 0, None)

        if update_state:
            # Refit with extended history
            y_train = np.asarray(y_test, dtype=float)
            self.fit_sequence(y_train)

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
        future = self._make_future(n_periods)
        fcst = self._model.predict(future)
        pred = fcst["yhat"].values
        if self.clip_negative:
            pred = np.clip(pred, 0, None)
        return pred.reshape(-1, 1, 1)

    def __repr__(self) -> str:
        return (
            f"ProphetForecaster(growth={self.growth}, "
            f"seasonality={self.seasonality_mode})"
        )
