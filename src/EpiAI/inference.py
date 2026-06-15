"""
InferencePipeline — run predictions on new data with a trained model.

Usage::

    # After training
    result = EpiAITrainer(model).fit(bundle)
    inferer = InferencePipeline.from_train_result(result)

    # Single-step prediction (torch / sklearn)
    pred = inferer.predict(new_df)          # (N, horizon, target_dim)

    # Pure forecast (ARIMA / ETS)
    forecast = inferer.forecast(steps=6)

    # Online update (ARIMA)
    updated = inferer.update(new_observations)

    # Persist
    inferer.save("model_package.zip")
    inferer = InferencePipeline.load("model_package.zip")
"""

from __future__ import annotations

import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from EpiAI.dataset.base import Compose
from EpiAI.dataset.transforms import SlidingWindow
from EpiAI.models.base import BaseForecaster
from EpiAI.models.registry import list_models
from EpiAI.trainer import TrainResult


# =====================================================================
# InferencePipeline
# =====================================================================

class InferencePipeline:
    """Deploy a trained model to make predictions on new data.

    Parameters
    ----------
    model : BaseForecaster
        Trained model instance.
    transforms : Compose or None
        Fitted transform pipeline (may be None if no transforms were used).
    lookback : int
        Number of past time steps used by the model.
    horizon : int
        Number of future time steps predicted.
    feature_names : list of str
        Names of input feature columns.
    target_names : list of str
        Names of target columns.
    paradigm : {"torch", "sklearn", "ts"}
        Model paradigm.
    """

    def __init__(
        self,
        model: BaseForecaster,
        transforms: Optional[Compose],
        lookback: int,
        horizon: int,
        feature_names: List[str],
        target_names: List[str],
        paradigm: str,
    ) -> None:
        self.model = model
        self.transforms = transforms
        self.lookback = lookback
        self.horizon = horizon
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.paradigm = paradigm

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_train_result(cls, result: TrainResult) -> "InferencePipeline":
        """Build from a TrainResult.

        Requires that the original ``PipelineBundle`` was attached to
        the result (via ``result._bundle`` or similar).  If not, use
        ``from_components()`` or pass bundle explicitly.
        """
        bundle = getattr(result, "_bundle", None)
        if bundle is None:
            raise ValueError(
                "TrainResult has no attached bundle. "
                "Use from_components() or pass bundle explicitly."
            )
        return cls.from_train_result_with_bundle(result, bundle)

    @classmethod
    def from_train_result_with_bundle(
        cls,
        result: TrainResult,
        bundle: Any,
    ) -> "InferencePipeline":
        """Build from a TrainResult + the original PipelineBundle."""
        return cls(
            model=result.model,
            transforms=bundle.transforms,
            lookback=bundle.lookback,
            horizon=bundle.horizon,
            feature_names=bundle.feature_names,
            target_names=bundle.target_names,
            paradigm=result.model.paradigm(),
        )

    @classmethod
    def from_components(
        cls,
        model: BaseForecaster,
        transforms: Optional[Compose] = None,
        lookback: int = 12,
        horizon: int = 3,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ) -> "InferencePipeline":
        """Build from individual components (for manual assembly)."""
        return cls(
            model=model,
            transforms=transforms,
            lookback=lookback,
            horizon=horizon,
            feature_names=feature_names or [],
            target_names=target_names or [],
            paradigm=model.paradigm(),
        )

    # ── Prediction ──────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on new data.

        Parameters
        ----------
        df : pd.DataFrame
            New data with the same columns as training data
            (must contain ``feature_names``).

        Returns
        -------
        np.ndarray
            Predictions, shape ``(N, horizon, target_dim)``,
            inverse-transformed to original scale.
        """
        if self.paradigm == "ts":
            raise TypeError(
                "TS-paradigm models do not support predict(df). "
                "Use forecast() or update() instead."
            )

        # 1. Apply transforms (no re-fitting)
        df_t = self._apply_transforms(df)

        # 2. Generate sliding windows (no target values needed during inference)
        sw = SlidingWindow(lookback=self.lookback, horizon=self.horizon)
        x_windows = sw.apply_features_only(df_t, self.feature_names)

        if len(x_windows) == 0:
            raise ValueError(
                f"Input has {len(df)} rows, but need at least "
                f"{self.lookback} rows to make one window."
            )

        # 3. Model prediction
        raw_pred = self.model.predict(x_windows)
        raw_pred = np.asarray(raw_pred, dtype=np.float32)

        # 4. Inverse transform predictions back to original scale
        return self._inverse_target(raw_pred)

    def forecast(self, steps: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Pure forecast into unknown future (TS-paradigm models only).

        Parameters
        ----------
        steps : int
            Number of time steps to forecast.
        X_future : pd.DataFrame or None, optional
            Future exogenous features (if the model supports them).

        Returns
        -------
        np.ndarray
            Forecast, shape ``(steps, horizon, target_dim)``.
        """
        if self.paradigm != "ts":
            raise TypeError(
                "forecast() is only for TS-paradigm models. "
                "Use predict() for window-based models."
            )
        X_arr = X_future.values if X_future is not None else None
        if X_arr is not None:
            raw = self.model.forecast(steps, X_arr)
        else:
            raw = self.model.forecast(steps)
        raw = np.asarray(raw, dtype=np.float32)
        while raw.ndim < 3:
            raw = np.expand_dims(raw, -1)
        return self._inverse_target(raw)

    def update(
        self,
        y_new: np.ndarray,
        X_new: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Online update with new observations (TS-paradigm only).

        Uses the model's ``predict_sequence()`` with ``update_state=True``
        to incorporate new observations and update internal state.

        Parameters
        ----------
        y_new : np.ndarray
            Newly observed target values, shape ``(T,)`` or ``(T, 1)``.
        X_new : np.ndarray or None, optional
            Corresponding exogenous features.

        Returns
        -------
        np.ndarray
            Updated predictions for the observed period.
        """
        if self.paradigm != "ts":
            raise TypeError("update() is only for TS-paradigm models.")
        y_new = np.asarray(y_new, dtype=np.float32).ravel()
        raw = self.model.predict_sequence(y_new, X_new, update_state=True)
        # Handle DataFrame output (ETS returns DataFrame with y_pred column)
        if isinstance(raw, pd.DataFrame):
            raw = raw["y_pred"].values
        raw = np.asarray(raw, dtype=np.float32)
        while raw.ndim < 3:
            raw = np.expand_dims(raw, -1)
        return self._inverse_target(raw)

    # ── Serialisation ───────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> str:
        """Save the inference pipeline to a zip archive.

        The archive contains::

            config.json        — metadata (lookback, horizon, …)
            model.pkl          — pickled model
            transforms.pkl     — pickled transforms (or empty)

        Parameters
        ----------
        path : str or Path
            Output path (``.zip`` extension recommended).

        Returns
        -------
        str
            The absolute path of the saved file.
        """
        path = Path(path)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")

        config = {
            "paradigm": self.paradigm,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "model_class": f"{type(self.model).__module__}.{type(self.model).__name__}",
        }

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("config.json", json.dumps(config, indent=2))
            zf.writestr("model.pkl", pickle.dumps(self.model))
            zf.writestr(
                "transforms.pkl",
                pickle.dumps(self.transforms) if self.transforms else b"",
            )

        resolved = str(path.resolve())
        return resolved

    @classmethod
    def load(cls, path: Union[str, Path]) -> "InferencePipeline":
        """Load an inference pipeline from a zip archive.

        Parameters
        ----------
        path : str or Path
            Path to the ``.zip`` archive produced by ``save()``.

        Returns
        -------
        InferencePipeline
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        with zipfile.ZipFile(path, "r") as zf:
            config = json.loads(zf.read("config.json"))
            model = pickle.loads(zf.read("model.pkl"))
            transforms_raw = zf.read("transforms.pkl")
            transforms = pickle.loads(transforms_raw) if transforms_raw else None

        return cls(
            model=model,
            transforms=transforms,
            lookback=config["lookback"],
            horizon=config["horizon"],
            feature_names=config["feature_names"],
            target_names=config["target_names"],
            paradigm=config["paradigm"],
        )

    # ── Internal helpers ────────────────────────────────────────

    def _apply_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transforms to a new DataFrame.

        Only ``transform()`` is called — never ``fit()``.
        """
        if self.transforms is None:
            return df
        return self.transforms.transform(df)

    def _inverse_target(self, preds: np.ndarray) -> np.ndarray:
        """Inverse-transform prediction columns only.

        Works column-by-column to avoid transform mismatches
        (scalers fitted on features may not have the same columns).
        """
        if self.transforms is None:
            return preds

        preds = preds.copy()
        n, h, t = preds.shape
        for i, tn in enumerate(self.target_names):
            col_pred = f"{tn}_pred"
            inv_df = pd.DataFrame(
                np.column_stack([preds[:, -1, i], preds[:, -1, i]]),
                columns=[tn, col_pred],
            )
            try:
                inv_df = self.transforms.inverse(inv_df)
                preds[:, -1, i] = inv_df[col_pred].values
            except Exception:
                pass  # keep raw value if inverse fails
        return preds

    def __repr__(self) -> str:
        return (
            f"InferencePipeline(paradigm={self.paradigm}, "
            f"model={type(self.model).__name__}, "
            f"lookback={self.lookback}, horizon={self.horizon}, "
            f"features={len(self.feature_names)}, targets={len(self.target_names)})"
        )


__all__ = ["InferencePipeline"]
