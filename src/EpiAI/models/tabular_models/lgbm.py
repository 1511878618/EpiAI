import numpy as np
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor


class LGBMSingleForecaster:
    """
    单模型 LightGBM 时间序列预测器（带 SHAP 可解释性）

    输入输出：
    - 输入:  (N, lookback, input_dim)
    - 输出:  (N, horizon, target_dim)

    特点：
    - 单模型同时预测 horizon * target_dim
    - 自动构造时序特征名
    - 内置 SHAP 全局 / 局部解释
    """

    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        target_dim: int = 1,
        input_feature_names: list[str] | None = None,
        lgbm_params: dict | None = None,
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

        if lgbm_params is None:
            lgbm_params = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
            )

        self.model = LGBMRegressor(**lgbm_params)
        self.explainer = None

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

    def fit(self, x, y):
        """
        参数：
        - x: (N, lookback, input_dim)
        - y: (N, horizon) 或 (N, horizon, target_dim)
        """
        X = self._flatten_x(x)
        y = self._prepare_y(y)

        y_flat = y.reshape(y.shape[0], -1)
        self.model.fit(X, y_flat)

        self.explainer = shap.Explainer(self.model)

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

    def explain(self, x):
        """
        返回 SHAP explanation 对象。
        """
        if self.explainer is None:
            raise RuntimeError("Explainer is not initialized. Please call fit() first.")

        X = self._flatten_x(x)
        explanation = self.explainer(X)
        return explanation

    def plot_shap_summary(
        self,
        x,
        horizon_idx: int = 0,
        target_idx: int = 0,
        max_display: int = 20,
    ):
        """
        全局 SHAP beeswarm summary plot.
        """
        explanation = self.explain(x)
        output_idx = self._prepare_output_index(horizon_idx, target_idx)

        if explanation.values.ndim == 3:
            values = explanation.values[:, :, output_idx]
        else:
            values = explanation.values

        shap.summary_plot(
            values,
            explanation.data,
            feature_names=self.flatten_feature_names,
            max_display=max_display,
            show=True,
        )

    def plot_shap_bar(
        self,
        x,
        horizon_idx: int = 0,
        target_idx: int = 0,
        max_display: int = 20,
    ):
        """
        全局 SHAP bar plot.
        """
        explanation = self.explain(x)
        output_idx = self._prepare_output_index(horizon_idx, target_idx)

        if explanation.values.ndim == 3:
            values = explanation.values[:, :, output_idx]
        else:
            values = explanation.values

        shap.summary_plot(
            values,
            explanation.data,
            feature_names=self.flatten_feature_names,
            plot_type="bar",
            max_display=max_display,
            show=True,
        )

    def plot_shap_waterfall(
        self,
        x,
        sample_idx: int = 0,
        horizon_idx: int = 0,
        target_idx: int = 0,
        max_display: int = 20,
    ):
        """
        单样本局部解释图。
        """
        explanation = self.explain(x)
        output_idx = self._prepare_output_index(horizon_idx, target_idx)

        if explanation.values.ndim == 3:
            shap_exp = shap.Explanation(
                values=explanation.values[sample_idx, :, output_idx],
                base_values=explanation.base_values[sample_idx, output_idx]
                if np.ndim(explanation.base_values) > 1
                else explanation.base_values[sample_idx],
                data=explanation.data[sample_idx],
                feature_names=self.flatten_feature_names,
            )
        else:
            shap_exp = shap.Explanation(
                values=explanation.values[sample_idx],
                base_values=explanation.base_values[sample_idx]
                if np.ndim(explanation.base_values) > 0
                else explanation.base_values,
                data=explanation.data[sample_idx],
                feature_names=self.flatten_feature_names,
            )

        shap.plots.waterfall(shap_exp, max_display=max_display)
        plt.show()

    def get_global_importance(
        self,
        x,
        horizon_idx: int = 0,
        target_idx: int = 0,
    ):
        """
        返回某个输出维度上的全局平均 |SHAP| 重要性表。
        """
        explanation = self.explain(x)
        output_idx = self._prepare_output_index(horizon_idx, target_idx)

        if explanation.values.ndim == 3:
            values = explanation.values[:, :, output_idx]
        else:
            values = explanation.values

        importance = np.abs(values).mean(axis=0)

        return [
            {"feature": feat, "importance": float(score)}
            for feat, score in sorted(
                zip(self.flatten_feature_names, importance),
                key=lambda z: z[1],
                reverse=True,
            )
        ]
