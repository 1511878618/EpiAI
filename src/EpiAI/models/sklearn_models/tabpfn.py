import numpy as np
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    _HAS_TABPFN = False
else:
    _HAS_TABPFN = True



from EpiAI.models.base import SklearnMixin
from EpiAI.models.registry import register

@register("TabPFN")
class TabPFNMultiForecaster(SklearnMixin):
    """
    多模型 TabPFN 时间序列预测器

    特点：
    - 每个 (horizon, target_dim) 一个 TabPFNRegressor
    - 更接近 direct multi-step forecasting
    - 训练更慢，但通常比单模型策略更稳
    """

    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        target_dim: int = 1,
        model_path: str | None = None,
        tabpfn_kwargs: dict | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        if tabpfn_kwargs is None:
            tabpfn_kwargs = {}

        self.models = []
        for _h in range(horizon):
            row = []
            for _t in range(target_dim):
                if model_path is not None:
                    model = TabPFNRegressor(model_path=model_path, **tabpfn_kwargs)
                else:
                    model = TabPFNRegressor(**tabpfn_kwargs)
                row.append(model)
            self.models.append(row)

    def _flatten_x(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(x.shape[0], -1)

    def _prepare_y(self, y):
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 2:
            y = y[..., None]

        if y.ndim != 3:
            raise ValueError(f"`y` must have shape (N, horizon) or (N, horizon, target_dim), got {y.shape}.")

        return y

    def fit(self, x, y):
        X = self._flatten_x(x)
        y = self._prepare_y(y)

        for h in range(self.horizon):
            for t in range(self.target_dim):
                self.models[h][t].fit(X, y[:, h, t])

    def predict(self, x):
        X = self._flatten_x(x)
        N = X.shape[0]

        preds = np.zeros((N, self.horizon, self.target_dim), dtype=np.float32)

        for h in range(self.horizon):
            for t in range(self.target_dim):
                preds[:, h, t] = self.models[h][t].predict(X)

        return preds
