"""
InferencePipeline — run predictions on new data with a trained model.
ModelVault — store, compare, and deploy multiple trained models.

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

    # Multi-model vault
    vault = ModelVault.from_results({"RF": result_rf, "XGB": result_xgb}, bundle)
    vault.save("/tmp/dengue_vault/")
    vault.summary()                         # comparison table
    vault.predict_all(new_data)             # all models at once
"""

from __future__ import annotations

import json
import pickle
import shutil
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
        """Inverse-transform prediction columns using the shared helper."""
        from EpiAI.trainer import inverse_predictions as _inv

        return _inv(preds, self.target_names, self.transforms, y_true=None)

    def __repr__(self) -> str:
        return (
            f"InferencePipeline(paradigm={self.paradigm}, "
            f"model={type(self.model).__name__}, "
            f"lookback={self.lookback}, horizon={self.horizon}, "
            f"features={len(self.feature_names)}, targets={len(self.target_names)})"
        )


# =====================================================================
# ModelVault — multi-model store, compare & deploy
# =====================================================================

class ModelVault:
    """Store, compare, and deploy multiple trained models.

    Directory structure::

        /path/to/vault/
        ├── manifest.json            # all models with metrics
        ├── RF/
        │   ├── model.zip            # InferencePipeline package
        │   └── meta.json            # training parameters
        └── ...

    Parameters
    ----------
    models : dict of str → InferencePipeline
        Name → inference pipeline map.
    metrics : pd.DataFrame
        Comparison table (model × MAE/RMSE/R²/PearsonR/n).
    bundle : PipelineBundle
        The original data bundle used for training.
    """

    def __init__(
        self,
        models: Dict[str, InferencePipeline],
        metrics: pd.DataFrame,
        bundle: Any,
    ) -> None:
        self.models = models
        self.metrics = metrics
        self.bundle = bundle

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_results(
        cls,
        results: Dict[str, "TrainResult"],
        bundle: Any,
    ) -> "ModelVault":
        """Build from a dict of ``{model_name: TrainResult}``.

        Each result gets an attached ``_bundle`` if not already set.
        """
        pipelines: Dict[str, InferencePipeline] = {}
        rows = []

        for name, result in results.items():
            if not hasattr(result, "_bundle") or result._bundle is None:
                result._bundle = bundle
            pip = InferencePipeline.from_train_result(result)
            pipelines[name] = pip

            m = result.metrics.iloc[0]
            rows.append({
                "model": name,
                "paradigm": result.model.paradigm(),
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "MAPE": m.get("MAPE", float("nan")),
                "R2": m["R2"],
                "PearsonR": m["PearsonR"],
                "n": m["n"],
            })

        metrics = pd.DataFrame(rows).set_index("model")
        return cls(pipelines, metrics, bundle)

    # ── Query ───────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return comparison table (sorted by R² descending)."""
        return self.metrics.sort_values("R2", ascending=False)

    def best(self, metric: str = "R2") -> str:
        """Return the name of the best model by *metric*.  Higher is better."""
        best_name = self.metrics[metric].idxmax()
        return best_name  # type: ignore[no-any-return]

    def get(self, name: str) -> InferencePipeline:
        """Get a specific model's inference pipeline."""
        return self.models[name]

    def __getitem__(self, name: str) -> InferencePipeline:
        return self.models[name]

    # ── Batch inference ────────────────────────────────────────

    def predict_all(
        self,
        new_data: Optional[pd.DataFrame] = None,
        steps: int = 6,
    ) -> Dict[str, np.ndarray]:
        """Run inference on all models and return ``{name: predictions}``.

        For window-based (torch/sklearn) models: pass ``new_data``
        with feature columns.  For TS models: pass ``steps``.
        """
        out: Dict[str, np.ndarray] = {}
        for name, pip in self.models.items():
            if pip.paradigm == "ts":
                out[name] = pip.forecast(steps)
            else:
                out[name] = pip.predict(new_data)
        return out

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> str:
        """Save all models to a vault directory.

        Parameters
        ----------
        path : str or Path
            Directory path (created if it doesn't exist).

        Returns
        -------
        str
            Absolute path to the vault directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for name, pip in self.models.items():
            model_dir = path / name
            model_dir.mkdir(exist_ok=True)
            pip.save(str(model_dir / "model.zip"))

            # Build training metadata
            meta = {
                "model": name,
                "paradigm": pip.paradigm,
                "model_class": f"{type(pip.model).__module__}.{type(pip.model).__name__}",
                "lookback": pip.lookback,
                "horizon": pip.horizon,
                "n_features": len(pip.feature_names),
                "n_targets": len(pip.target_names),
                "feature_names": pip.feature_names,
                "target_names": pip.target_names,
            }
            if name in self.metrics.index:
                row = self.metrics.loc[name]
                meta["metrics"] = {
                    k: float(v) if isinstance(v, (np.floating, float))
                    else int(v) if isinstance(v, (np.integer, int))
                    else v
                    for k, v in row.to_dict().items()
                    if k != "model"
                }
            (model_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False)
            )

        # Save manifest
        manifest = {
            "models": list(self.models.keys()),
            "n_models": len(self.models),
            "created": str(pd.Timestamp.now()),
        }

        # Metrics table as an ordered list of dicts
        manifest["metrics"] = {
            name: {
                k: float(v) if isinstance(v, (np.floating, float))
                else int(v) if isinstance(v, (np.integer, int))
                else v
                for k, v in row.to_dict().items()
            }
            for name, row in self.metrics.iterrows()
        }

        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )

        return str(path.resolve())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelVault":
        """Load a vault directory created by ``save()``."""
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())

        metrics = pd.DataFrame(manifest["metrics"]).T
        metrics.index.name = "model"

        models: Dict[str, InferencePipeline] = {}
        for name in manifest["models"]:
            model_zip = path / name / "model.zip"
            if model_zip.exists():
                models[name] = InferencePipeline.load(str(model_zip))

        return cls(models, metrics, bundle=None)

    def __repr__(self) -> str:
        return f"ModelVault({len(self.models)} models, {len(self.metrics.columns)} metrics)"


# =====================================================================
# DeploymentRuntime — production deployment with unified data table
# =====================================================================

class TimeGapError(ValueError):
    """Raised when new data creates a time gap."""

class TimeOrderError(ValueError):
    """Raised when new data is older than the latest record."""

class BufferError(RuntimeError):
    """Raised when data_table has fewer rows than a model's lookback."""


class DeploymentRuntime:
    """Production runtime that maintains a persistent data table.

    Each call to ``feed(new_data)`` appends data, checks time continuity,
    and runs predictions for all models in the vault.  TS models are not
    auto-updated — call ``update_model()`` explicitly.

    Parameters
    ----------
    vault : ModelVault
        Trained models.
    time_col : str
        Name of the time column.
    time_unit : str, optional
        Pandas offset alias (``\"MS\"``, ``\"D\"``, etc.).
    strict : bool, default=True
        When True, time gaps raise ``TimeGapError``.
    """

    def __init__(
        self,
        vault: ModelVault,
        time_col: str = "time",
        time_unit: str = "MS",
        strict: bool = True,
    ) -> None:
        self.vault = vault
        self.time_col = time_col
        self.time_unit = time_unit
        self.strict = strict
        self._time_delta = pd.tseries.frequencies.to_offset(time_unit)
        self._feed_count = 0
        self._path: Optional[Path] = None

        # data_table: shared persistent history
        self.data_table: pd.DataFrame = pd.DataFrame()

        # Infer train_end_time from the first model's meta
        self._train_end_time: Optional[pd.Timestamp] = None

    # ── Core feed ────────────────────────────────────────────────

    def feed(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Process new observations and return predictions.

        Parameters
        ----------
        new_data : pd.DataFrame
            One or more rows with ``time_col`` and feature/target columns.

        Returns
        -------
        dict of {name: {"time": DatetimeIndex, "pred": ndarray}}
        """
        new_data = new_data.copy()
        self._check_time_continuity(new_data)
        self.data_table = pd.concat(
            [self.data_table, new_data], ignore_index=True
        )
        self._feed_count += 1

        results: Dict[str, Any] = {}
        last_time = pd.to_datetime(self.data_table[self.time_col].iloc[-1])

        for name, inferer in self.vault.models.items():
            try:
                if inferer.paradigm == "ts":
                    # TS: forecast from current state (no auto-update).
                    # Each feed advances the prediction window:
                    #   feed 1: forecast(horizon)     → [t+1, t+2, t+3]
                    #   feed 2: forecast(horizon+1)   → take last 3 → [t+2, t+3, t+4]
                    #   feed 3: forecast(horizon+2)   → take last 3 → [t+3, t+4, t+5]
                    n_fcst = inferer.horizon + max(0, self._feed_count - 1)
                    raw = inferer.forecast(n_fcst)
                    preds = np.asarray(raw, dtype=np.float32).ravel()
                    # Take the last horizon steps that correspond to NOW + future
                    preds = preds[-inferer.horizon:]
                    future = pd.date_range(
                        start=last_time + self._time_delta,
                        periods=inferer.horizon,
                        freq=self._time_delta,
                    )
                    results[name] = {"time": future, "pred": preds}

                else:
                    # Window model: pull last lookback rows
                    lb = inferer.lookback
                    if len(self.data_table) < lb:
                        raise BufferError(
                            f"{name} needs {lb} rows, "
                            f"data_table has {len(self.data_table)}"
                        )
                    # Pass raw DataFrame slice — InferencePipeline handles
                    # transforms + windowing internally
                    x_df = self.data_table.tail(lb)[inferer.feature_names]
                    raw = inferer.predict(x_df)
                    future = pd.date_range(
                        start=last_time + self._time_delta,
                        periods=inferer.horizon,
                        freq=self._time_delta,
                    )
                    results[name] = {"time": future, "pred": raw[0, :, 0]}

            except Exception as e:
                results[name] = {"error": str(e)}

        self._persist()
        return results

    # ── Explicit TS model update ─────────────────────────────────

    def update_model(self, name: str, new_y: np.ndarray) -> None:
        """Explicitly update a TS model's internal state.

        This is a separate step from ``feed()`` — the user decides
        when data quality is sufficient for a state update.
        """
        inferer = self.vault.get(name)
        if inferer.paradigm != "ts":
            raise TypeError(f"{name} is not a TS model.")

        # Backup current state before updating
        if self._path is not None:
            backup_dir = self._path / "ts_backup" / name
            backup_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                backup_dir / f"y_history_{self._feed_count}.npy",
                inferer.model.y_history_,
            )

        inferer.update(np.asarray(new_y, dtype=np.float32))
        self._persist()

    def update_all_ts(self, data: pd.DataFrame) -> None:
        """Batch-update all TS models (reserved for future retrain)."""
        for name, inferer in self.vault.models.items():
            if inferer.paradigm == "ts":
                y_new = data[inferer.target_names].values.ravel().astype(np.float32)
                self.update_model(name, y_new)

    # ── Time continuity ──────────────────────────────────────────

    def _check_time_continuity(self, new_data: pd.DataFrame) -> None:
        new_times = pd.to_datetime(new_data[self.time_col])

        if self.data_table.empty:
            # First feed: must follow train_end_time
            if self._train_end_time is not None:
                expected = self._train_end_time + self._time_delta
                if new_times[0] != expected:
                    self._raise_or_warn(
                        TimeGapError,
                        f"First feed time {new_times[0]} does not follow "
                        f"train_end {self._train_end_time}. Expected {expected}.",
                    )
            elif self.strict:
                self._raise_or_warn(
                    TimeGapError,
                    "First feed: _train_end_time is not set. "
                    "Set runtime._train_end_time = training_data['time'].iloc[-1] "
                    "to enable time continuity checking.",
                )
            return

        last = pd.to_datetime(self.data_table[self.time_col].iloc[-1])

        # Check for disorder (new time <= last seen)
        if new_times[0] <= last:
            self._raise_or_warn(
                TimeOrderError,
                f"Time disorder: last={last}, new={new_times[0]}",
            )

        # Check each new row for continuity
        expected = last + self._time_delta
        for i, t in enumerate(new_times):
            expected_i = expected + i * self._time_delta
            if t != expected_i:
                self._raise_or_warn(
                    TimeGapError,
                    f"Time gap: last={last}, expected={expected_i}, "
                    f"got={t}. Missing {expected_i} ~ {t - self._time_delta}.",
                )

    def _raise_or_warn(self, exc_type, msg: str) -> None:
        if self.strict:
            raise exc_type(msg)
        import warnings
        warnings.warn(f"[DeploymentRuntime] {msg}", stacklevel=3)

    # ── Persistence ──────────────────────────────────────────────

    def _persist(self) -> None:
        """Auto-save after each feed/update (no-op if never saved)."""
        if self._path is not None:
            self.save(str(self._path))

    def _save_data_table(self, path: Path) -> None:
        """Save data_table. Prefers parquet, falls back to CSV."""
        if self.data_table.empty:
            return
        try:
            self.data_table.to_parquet(path / "data_table.parquet")
        except ImportError:
            self.data_table.to_csv(path / "data_table.csv", index=False)

    def save(self, path: Union[str, Path]) -> str:
        """Persist runtime state to disk."""
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

        self._path = path

        # data_table
        self._save_data_table(path)

        # vault
        self.vault.save(str(path / "vault"))

        # meta
        meta = {
            "feed_count": self._feed_count,
            "time_col": self.time_col,
            "time_unit": self.time_unit,
            "strict": self.strict,
            "train_end_time": str(self._train_end_time) if self._train_end_time else None,
            "last_time": str(self.data_table[self.time_col].iloc[-1])
            if not self.data_table.empty else None,
            "n_rows": len(self.data_table),
        }
        (path / "runtime_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

        return str(path.resolve())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeploymentRuntime":
        """Restore runtime state from disk."""
        path = Path(path)
        meta_path = path / "runtime_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Not a DeploymentRuntime directory: {path}")

        meta = json.loads(meta_path.read_text())
        vault = ModelVault.load(str(path / "vault"))

        runtime = cls(
            vault=vault,
            time_col=meta["time_col"],
            time_unit=meta["time_unit"],
            strict=meta["strict"],
        )
        runtime._feed_count = meta["feed_count"]
        runtime._path = path

        if meta["train_end_time"]:
            runtime._train_end_time = pd.Timestamp(meta["train_end_time"])

        # Restore data_table
        dt_path = path / "data_table.parquet"
        if dt_path.exists():
            runtime.data_table = pd.read_parquet(dt_path)
        else:
            csv_path = path / "data_table.csv"
            if csv_path.exists():
                runtime.data_table = pd.read_csv(csv_path)

        return runtime

    def __repr__(self) -> str:
        return (
            f"DeploymentRuntime({len(self.vault.models)} models, "
            f"{len(self.data_table)} rows, "
            f"feed_count={self._feed_count})"
        )


__all__ = ["InferencePipeline", "ModelVault", "DeploymentRuntime",
           "TimeGapError", "TimeOrderError", "BufferError"]
