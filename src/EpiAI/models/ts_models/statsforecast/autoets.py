"""
StatsForecast AutoETS forecaster — drop-in wrapper for EpiAI's TS paradigm.

Replaces the current statsmodels-based ``ETS`` with Nixtla's faster C++
implementation.  ``AutoETS`` automatically searches the error/trend/season
space (``model='ZZZ'``) to find the best ETS specification.

Registered name: ``"AutoETS"`` (distinct from the original ``"ETS"``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register

try:
    from statsforecast.models import AutoETS as _SF_AutoETS
except ImportError:
    _SF_AutoETS = None  # type: ignore[assignment]


@register("AutoETS")
class StatsAutoETS(TSMixin):
    """StatsForecast AutoETS — univariate, no exogenous support.

    Automatically selects the best ETS (Error, Trend, Seasonal)
    specification via information criterion.

    Parameters
    ----------
    season_length : int, default=12
        Number of periods per season (e.g. 12 for monthly data).
    model : str, default="ZZZ"
        Model string: ``"AAA"`` (all additive), ``"MAM"``, ``"ZZZ"``
        (auto-select), etc.  The three characters denote Error, Trend,
        and Seasonality respectively.
    clip_negative : bool, default=True
        Clip negative predictions to zero.
    **kwargs
        Extra parameters forwarded to ``statsforecast.models.AutoETS``.
    """

    # ── Capability declarations ──────────────────────────────────
    target_dim_supported = 1
    exogenous_supported = False

    def __init__(
        self,
        season_length: int = 12,
        model: str = "ZZZ",
        clip_negative: bool = True,
        **kwargs,
    ) -> None:
        if _SF_AutoETS is None:
            raise ImportError(
                "StatsForecast is required. Install with: pip install statsforecast"
            )

        self.season_length = season_length
        self.model = model
        self.clip_negative = clip_negative
        self._model_kwargs = kwargs

        # Internal state
        self._fitted: _SF_AutoETS | None = None
        self._last_ds: pd.Timestamp | None = None

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _to_1d(arr) -> np.ndarray:
        y = np.asarray(arr, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"Expected 1-D y, got shape {y.shape}")
        if np.isnan(y).any():
            raise ValueError("y contains NaN.")
        return y

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(pred, 0, None) if self.clip_negative else pred

    # ── TSMixin interface ────────────────────────────────────────

    def fit_sequence(self, y_train, X_train=None, **kwargs):
        """Fit AutoETS on training sequence (X_train is ignored)."""
        if X_train is not None:
            raise ValueError("AutoETS does not support exogenous variables.")

        y = self._to_1d(y_train)
        model = _SF_AutoETS(
            season_length=self.season_length,
            model=self.model,
            **self._model_kwargs,
        )
        model.fit(y)
        self._fitted = model
        return self

    def predict_sequence(self, y_test, X_test=None, **kwargs):
        """Forecast the entire test period in one shot."""
        if self._fitted is None:
            raise RuntimeError("Model not fitted. Call fit_sequence() first.")
        n_test = len(self._to_1d(y_test))
        raw = self._fitted.predict(h=n_test)
        return self._clip(np.asarray(raw["mean"], dtype=float))

    def forecast(self, n_periods: int, X_future=None):
        """Forecast truly unknown future values."""
        if self._fitted is None:
            raise RuntimeError("Model not fitted. Call fit_sequence() first.")
        raw = self._fitted.predict(h=n_periods)
        pred = self._clip(np.asarray(raw["mean"], dtype=float))
        return pred.reshape(-1, 1, 1)

    def __repr__(self) -> str:
        return f"StatsAutoETS(season_length={self.season_length}, model={self.model!r})"
