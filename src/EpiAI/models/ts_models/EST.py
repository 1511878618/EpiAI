try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
except ImportError:
    ETSModel = None
import numpy as np
import pandas as pd

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register


@register("ETS", "ets", "exponential_smoothing")
class ETSForecaster(TSMixin):
    """
    ETS (Error-Trend-Seasonality) forecaster.

    基于 statsmodels，单变量，不支持 X。
    一次性 fit 完整训练集，然后 forecast 未来 horizon 步。
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        error: str = 'add',
        trend: str | None = 'add',
        seasonal: str | None = 'add',
        damped_trend: bool = False,
        clip_negative: bool = True,
        **kwagrs
    ):
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.clip_negative = clip_negative

        self._result = None
        self._y_train = None

    def fit_sequence(self, y_train, X_train=None):
        """一次性拟合 ETS 模型，不支持 X。"""
        if X_train is not None:
            raise ValueError("ETS does not support exogenous variables (X_train).")

        y_train = np.asarray(y_train, dtype=float)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train[:, 0]
        if y_train.ndim != 1:
            raise ValueError(f"y_train must be 1D, got {y_train.shape}")

        self._y_train = y_train

        model = ETSModel(
            y_train,
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal if self.seasonal else None,
            seasonal_periods=self.seasonal_periods if self.seasonal else None,
            damped_trend=self.damped_trend,
        )
        self._result = model.fit(disp=False)
        return self

    def predict_sequence(self, y_test, X_test=None, update_state=True, return_df=True):
        """在训练集上直接 forecast horizon 步，与 y_test 比较。"""
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if X_test is not None:
            raise ValueError("ETS does not support exogenous variables (X_test).")

        y_test = np.asarray(y_test, dtype=float)
        if y_test.ndim == 2 and y_test.shape[1] == 1:
            y_test = y_test[:, 0]

        horizon = len(y_test)
        pred = self._result.forecast(steps=horizon)
        pred = np.asarray(pred, dtype=float)

        if self.clip_negative:
            pred = np.clip(pred, 0, None)

        if not return_df:
            return pred

        return pd.DataFrame({
            "time_index": np.arange(horizon),
            "y_true": y_test,
            "y_pred": pred,
            "abs_error": np.abs(y_test - pred),
        })

    def forecast(self, n_periods: int):
        """对未来未知值做纯预测（无 y_test）。"""
        if self._result is None:
            raise RuntimeError("Model not fitted.")
        pred = self._result.forecast(steps=n_periods)
        pred = np.asarray(pred, dtype=float)
        if self.clip_negative:
            pred = np.clip(pred, 0, None)
        return pred
