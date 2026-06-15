from __future__ import annotations

import numpy as np
import pandas as pd
try:
    import pmdarima as pm
except ImportError:
    pm = None

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register


@register("ARIMA", "auto_arima", "pmdarima")
class AutoARIMAXRollingForecaster(TSMixin):
    """
    Auto-ARIMA / ARIMAX / SARIMAX rolling-origin forecaster.
    Note: SARIMAX 会需要传入 X_train

    Core logic
    ----------
    1. fit(y_train, X_train):
       - use auto_arima on training data to select order / seasonal_order
       - save training history

    2. predict(y_test, X_test):
       - rolling-origin prediction on test data
       - rolling_step = horizon（固定，无空洞无重叠）
       - use true y_test to update temporary history
       - does NOT update internal model history

    3. fit_predict(y_test, X_test):
       - same rolling-origin prediction as predict
       - updates internal history after prediction

    4. forecast(X_future):
       - forecast unknown future without future y
       - no rolling update because future y is unknown

    Parameters
    ----------
    seasonal : bool
        Whether to use seasonal ARIMA.
    m : int
        Seasonal period. For monthly data, use m=12.
    rolling_window_size : int, "all", or None
        - None: use initial training size
        - int: fixed-length sliding window
        - "all": expanding window using all available history
    horizon : int
        Forecast horizon at each rolling origin. Also used as rolling_step.
    clip_negative : bool
        Whether to clip negative predictions to zero.
    auto_arima_kwargs : dict
        Extra kwargs passed to pmdarima.auto_arima.
    verbose : bool
        Whether to print progress.
    """

    def __init__(
        self,
        seasonal: bool = True,
        m: int = 12,
        rolling_window_size: int | str | None = 'all',
        horizon: int = 1,
        clip_negative: bool = True,
        auto_arima_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> None:
        self.seasonal = seasonal
        self.m = m
        self.rolling_window_size = rolling_window_size
        self.horizon = horizon
        self.clip_negative = clip_negative
        self.auto_arima_kwargs = auto_arima_kwargs or {}
        self.verbose = verbose

        self.best_order_ = None
        self.best_seasonal_order_ = None
        self.search_model_ = None

        self.y_history_ = None
        self.X_history_ = None
        self.train_size_ = None
        self.n_features_ = None

        self.last_predictions_ = None
        self.last_metrics_ = None

    def _to_numpy(self, arr):
        if arr is None:
            return None
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _prepare_y(self, y) -> np.ndarray:
        y = self._to_numpy(y).astype(float)

        if y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError("Only target_dim=1 is supported.")
            y = y[:, 0]

        if y.ndim != 1:
            raise ValueError(f"y must have shape (T,) or (T, 1), got {y.shape}")

        if np.isnan(y).any():
            raise ValueError("y contains NaN. Please impute or remove missing target values.")

        return y

    def _prepare_X(self, X, T: int) -> np.ndarray | None:
        if X is None:
            return None

        X = self._to_numpy(X).astype(float)

        if X.ndim != 2:
            raise ValueError(f"X must have shape (T, n_features), got {X.shape}")

        if X.shape[0] != T:
            raise ValueError(f"X and y length mismatch: X={X.shape[0]}, y={T}")

        if np.isnan(X).any():
            raise ValueError("X contains NaN. Please impute or remove missing feature values.")

        return X

    def _check_is_fitted(self):
        if self.best_order_ is None or self.y_history_ is None:
            raise RuntimeError("Model is not fitted. Call fit(y_train, X_train) first.")

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        if self.clip_negative:
            pred = np.clip(pred, 0, None)
        return pred

    def _metrics(self, y_true, y_pred) -> dict:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

        return {
            "MAE": float(np.mean(np.abs(y_true - y_pred))),
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "MAPE": float(
                np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
            ),
        }

    def _resolve_window_start(self, origin: int) -> int:
        if self.rolling_window_size is None:
            window_size = self.train_size_
            return max(0, origin - window_size)

        if self.rolling_window_size == "all":
            return 0

        if isinstance(self.rolling_window_size, int):
            if self.rolling_window_size <= 0:
                raise ValueError("rolling_window_size must be positive.")
            return max(0, origin - self.rolling_window_size)

        raise ValueError("rolling_window_size must be None, int, or 'all'.")

    def _fit_fixed_order_model(self, y_window, X_window=None):
        model = pm.ARIMA(
            order=self.best_order_,
            seasonal_order=self.best_seasonal_order_,
            suppress_warnings=True,
        )
        model.fit(y_window, X=X_window)
        return model

    def fit_sequence(self, y_train, X_train=None):
        """
        Select ARIMA/SARIMAX order on training data and save training history.
        """
        y_train = self._prepare_y(y_train)
        X_train = self._prepare_X(X_train, len(y_train))

        self.train_size_ = len(y_train)
        self.n_features_ = None if X_train is None else X_train.shape[1]

        kwargs = {
            "seasonal": self.seasonal,
            "m": self.m,
            "suppress_warnings": True,
            "error_action": "ignore",
            "stepwise": True,
            "trace": self.verbose,
        }
        kwargs.update(self.auto_arima_kwargs)

        self.search_model_ = pm.auto_arima(
            y_train,
            X=X_train,
            **kwargs,
        )

        self.best_order_ = self.search_model_.order
        self.best_seasonal_order_ = self.search_model_.seasonal_order

        self.y_history_ = y_train.copy()
        self.X_history_ = None if X_train is None else X_train.copy()

        if self.verbose:
            print("Best order:", self.best_order_)
            print("Best seasonal_order:", self.best_seasonal_order_)

        return self

    def predict(self, y_test, X_test=None, return_df: bool = True):
        """
        Rolling-origin prediction on test data without updating internal history.

        rolling_step = horizon（固定），每次滚动刚好覆盖不重叠。

        y_test and X_test must immediately follow the training history.
        """
        self._check_is_fitted()

        y_test = self._prepare_y(y_test)
        X_test = self._prepare_X(X_test, len(y_test))

        if self.X_history_ is not None and X_test is None:
            raise ValueError("Model was fitted with X_train, so X_test must be provided.")

        if self.X_history_ is None and X_test is not None:
            raise ValueError("Model was fitted without X_train, so X_test should be None.")

        if X_test is not None and X_test.shape[1] != self.n_features_:
            raise ValueError(
                f"X_test feature mismatch: expected {self.n_features_}, got {X_test.shape[1]}"
            )

        y_all = np.concatenate([self.y_history_, y_test])
        X_all = None
        if self.X_history_ is not None:
            X_all = np.vstack([self.X_history_, X_test])

        T_train = len(self.y_history_)
        T_all = len(y_all)

        preds = np.full(len(y_test), np.nan, dtype=float)

        origin = T_train

        while origin < T_all:
            pred_end = min(origin + self.horizon, T_all)

            window_start = self._resolve_window_start(origin)

            y_window = y_all[window_start:origin]
            X_window = None if X_all is None else X_all[window_start:origin]
            X_future = None if X_all is None else X_all[origin:pred_end]

            try:
                model = self._fit_fixed_order_model(y_window, X_window)
                pred = model.predict(
                    n_periods=pred_end - origin,
                    X=X_future,
                )
                pred = self._clip(pred)

                local_start = origin - T_train
                local_end = pred_end - T_train
                preds[local_start:local_end] = pred

            except Exception as e:
                if self.verbose:
                    print(f"[rolling prediction error] origin={origin}: {e}")

            origin += self.horizon   # ← 唯一改动：rolling_step → horizon

        self.last_predictions_ = preds
        self.last_metrics_ = self._metrics(y_test, preds)

        if not return_df:
            return preds

        return pd.DataFrame(
            {
                "time_index": np.arange(len(y_test)),
                "y_true": y_test,
                "y_pred": preds,
                "abs_error": np.abs(y_test - preds),
            }
        )

    def predict_sequence(self, y_test, X_test=None, update_state: bool = True,
                         return_df: bool = True):
        """
        Rolling-origin prediction and then update internal history with true y_test/X_test.
        Only works when  rolling_window_size == 'all'
        """
        pred = self.predict(y_test, X_test, return_df=return_df)

        y_test_np = self._prepare_y(y_test)
        X_test_np = self._prepare_X(X_test, len(y_test_np))

        self.y_history_ = np.concatenate([self.y_history_, y_test_np])

        if self.X_history_ is not None:
            self.X_history_ = np.vstack([self.X_history_, X_test_np])

        return pred

    def forecast(self, n_periods: int, X_future=None):
        """
        Forecast truly unknown future values.

        This is not rolling-origin evaluation because future y is unavailable.
        """
        self._check_is_fitted()

        if n_periods <= 0:
            raise ValueError("n_periods must be positive.")

        X_future = self._prepare_X(X_future, n_periods)

        if self.X_history_ is not None and X_future is None:
            raise ValueError("Model was fitted with X, so X_future must be provided.")

        if self.X_history_ is None and X_future is not None:
            raise ValueError("Model was fitted without X, so X_future should be None.")

        model = self._fit_fixed_order_model(self.y_history_, self.X_history_)

        pred = model.predict(
            n_periods=n_periods,
            X=X_future,
        )

        return self._clip(pred)

    def summary(self) -> dict:
        return {
            "best_order": self.best_order_,
            "best_seasonal_order": self.best_seasonal_order_,
            "train_size": self.train_size_,
            "n_features": self.n_features_,
            "rolling_window_size": self.rolling_window_size,
            "horizon": self.horizon,
            "last_metrics": self.last_metrics_,
        }
