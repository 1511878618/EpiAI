"""
SVM-based single-model time series forecaster.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


from EpiAI.models.base import SklearnMixin
from EpiAI.models.registry import register

@register("SVR")
class SVRForecaster(SklearnMixin):
    """
    单模型 SVM 时间序列预测器。

    输入输出：
    - 输入:  (N, lookback, input_dim)
    - 输出:  (N, horizon, target_dim)

    特点：
    - 单模型同时预测 horizon * target_dim
    - 自动构造时序特征名
    - 内部使用 MultiOutputRegressor 包装 SVR
    """

    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        target_dim: int = 1,
        input_feature_names: list[str] | None = None,
        svm_params: dict | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        if input_feature_names is None:
            input_feature_names = [f"feat_{i}" for i in range(input_dim)]

        if len(input_feature_names) != input_dim:
            raise ValueError(
                f"`input_feature_names` length must equal input_dim={input_dim}, "
                f"but got {len(input_feature_names)}."
            )

        self.input_feature_names = input_feature_names
        self.flatten_feature_names = self._build_flatten_feature_names()

        if svm_params is None:
            svm_params = dict(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
                gamma="scale",
                tol=1e-3,
                max_iter=-1,
            )

        base_estimator = SVR(**svm_params)
        self.model = MultiOutputRegressor(base_estimator, n_jobs=-1)

    def _build_flatten_feature_names(self) -> list[str]:
        """
        构造 flatten 后的时序特征名。

        例如：
        lookback=3, input_feature_names=["temp", "rain"]
        =>
        [
            "lag_2_temp", "lag_2_rain",
            "lag_1_temp", "lag_1_rain",
            "lag_0_temp", "lag_0_rain",
        ]
        """
        names = []
        for t in range(self.lookback):
            lag = self.lookback - 1 - t
            for feat_name in self.input_feature_names:
                names.append(f"lag_{lag}_{feat_name}")
        return names

    def _flatten_x(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 3:
            raise ValueError(
                f"`x` must have shape (N, lookback, input_dim), got {x.shape}."
            )

        if x.shape[1] != self.lookback:
            raise ValueError(
                f"lookback mismatch: expected {self.lookback}, got {x.shape[1]}."
            )

        if x.shape[2] != self.input_dim:
            raise ValueError(
                f"input_dim mismatch: expected {self.input_dim}, got {x.shape[2]}."
            )

        return x.reshape(x.shape[0], -1)

    def _prepare_y(self, y):
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 2:
            y = y[..., None]

        if y.ndim != 3:
            raise ValueError(
                f"`y` must have shape (N, horizon) or (N, horizon, target_dim), got {y.shape}."
            )

        if y.shape[1] != self.horizon:
            raise ValueError(
                f"horizon mismatch: expected {self.horizon}, got {y.shape[1]}."
            )

        if y.shape[2] != self.target_dim:
            raise ValueError(
                f"target_dim mismatch: expected {self.target_dim}, got {y.shape[2]}."
            )

        return y

    def _prepare_output_index(self, horizon_idx=0, target_idx=0) -> int:
        if not (0 <= horizon_idx < self.horizon):
            raise ValueError(
                f"`horizon_idx` must be in [0, {self.horizon - 1}], got {horizon_idx}."
            )
        if not (0 <= target_idx < self.target_dim):
            raise ValueError(
                f"`target_idx` must be in [0, {self.target_dim - 1}], got {target_idx}."
            )
        return horizon_idx * self.target_dim + target_idx

    def fit(self, x, y, val_x=None, val_y=None, verbose=False):
        """
        参数：
        - x: (N, lookback, input_dim)
        - y: (N, horizon) 或 (N, horizon, target_dim)

        注：SVR 不支持 eval_set，val_x / val_y 保留仅用于接口一致性。
        """
        X = self._flatten_x(x)
        y = self._prepare_y(y)
        y_flat = y.reshape(y.shape[0], -1)

        self.model.fit(X, y_flat)

    def predict(self, x):
        """
        返回：
        - preds: (N, horizon, target_dim)
        """
        X = self._flatten_x(x)
        preds = self.model.predict(X)
        preds = np.asarray(preds, dtype=np.float32)

        if preds.ndim == 1:
            preds = preds[:, None]

        return preds.reshape(-1, self.horizon, self.target_dim)

    def get_flatten_feature_names(self) -> list[str]:
        return self.flatten_feature_names
