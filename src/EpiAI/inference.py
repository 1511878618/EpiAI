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
        # 记录模型 init 参数，供 retrain 时重建模型实例使用
        self.model_config: dict = {}

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

    def forecast(self, steps: int, X_future: Optional[pd.DataFrame] = None,
                 level: Optional[int] = None) -> np.ndarray:
        """Pure forecast into unknown future (TS-paradigm models only).

        Parameters
        ----------
        steps : int
            Number of time steps to forecast.
        X_future : pd.DataFrame or None, optional
            Future exogenous features (if the model supports them).
        level : int or None, optional
            Confidence level for prediction intervals (e.g. 95).
            When set, ``self.model._last_ci`` is populated after the call.

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
            raw = self.model.forecast(steps, X_arr, level=level)
        else:
            raw = self.model.forecast(steps, level=level)
        raw = np.asarray(raw, dtype=np.float32)
        while raw.ndim < 3:
            raw = np.expand_dims(raw, -1)

        # Inverse-transform CI bounds if present
        ci = getattr(self.model, "_last_ci", None)
        if ci is not None:
            for key in ("lower", "upper"):
                arr = np.asarray(ci[key], dtype=np.float32).reshape(-1, 1, 1)
                inv = self._inverse_target(arr)[:, 0, 0]
                ci[key] = inv
            self.model._last_ci = ci

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
        """Return comparison table (sorted by Pearson r descending)."""
        return self.metrics.sort_values("PearsonR", ascending=False)

    def best(self, metric: str = "PearsonR") -> str:
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
                "model_config": pip.model_config,
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
    """生产部署运行时，管理 data_table + vault 的预测与更新。

    核心设计
    --------
    - ``data_table``: 全部历史观测数据，**原始尺度**（未归一化），按时间排序
    - ``predict(horizon)``: 预测未来 horizon 步，返回 时间×模型 表格
    - ``feed(new_data)``: 追加新观测

    注意:
    - ``data_table`` 必须存储**原始尺度**数据（与训练时 CSV 一致）。
      模型内部 ``InferencePipeline`` 会自动处理 transform/inverse_transform。
    - ``predict()`` 返回的结果也是原始尺度。

    Parameters
    ----------
    vault : ModelVault
        已训练的模型集合。
    time_col : str
        时间列名。
    time_unit : str
        时间频率（\"MS\" 月, \"D\" 日, 默认 \"MS\"）。
    strict : bool
        严格模式，默认 True 时时间不连续报错。
    data_table : pd.DataFrame or None
        初始历史数据表（原始尺度）。可在之后通过 ``.data_table`` 属性设置。
    """

    def __init__(
        self,
        vault: ModelVault,
        time_col: str = "time",
        time_unit: str = "MS",
        strict: bool = True,
        data_table: Optional[pd.DataFrame] = None,
    ) -> None:
        self.vault = vault
        self.time_col = time_col
        self.time_unit = time_unit
        self.strict = strict
        self._time_delta = pd.tseries.frequencies.to_offset(time_unit)
        self._path: Optional[Path] = None
        self.data_table: pd.DataFrame = data_table.copy() if data_table is not None else pd.DataFrame()
        self.risk_scorer: Optional[object] = None  # set externally by ForecastTask
        self._history_end_time: Optional[pd.Timestamp] = (
            pd.to_datetime(self.data_table[self.time_col].iloc[-1])
            if not self.data_table.empty else None
        )

    def _validate_time_granularity(self):
        """检查 data_table 时间列颗粒度与 time_unit 一致，并检查缺失值。"""
        if self.data_table.empty:
            return
        times = pd.to_datetime(self.data_table[self.time_col])
        # 检查 NA
        if times.isna().any():
            print("[WARN] data_table 时间列存在缺失值 (NaT).")
        # 检查颗粒度：相邻时间差是否都是 time_unit 的整数倍
        diffs = times.diff().dropna()
        expected = pd.Timedelta(self._time_delta)
        bad = diffs[diffs != expected]
        if not bad.empty:
            msg = (f"时间颗粒度不一致: 期望 {expected}, "
                   f"发现差异 {bad.unique().tolist()}")
            if self.strict:
                raise TimeGapError(msg)
            else:
                print(f"[WARN] {msg}")

    # ── Core: predict ─────────────────────────────────────────

    def predict(self, horizon: int = 1, level: Optional[int] = None) -> pd.DataFrame:
        """预测未来 horizon 步，返回 时间×模型 表格。

        Parameters
        ----------
        horizon : int
            预测步数，默认 1。
        level : int or None, optional
            置信水平（如 95）。支持此参数的模型会在原预测值旁
            追加 ``{model}_lo`` 和 ``{model}_hi`` 列。

        Returns
        -------
        pd.DataFrame
            行 = 时间点, 列 = 模型名（以及风险/CI 列）。
        """
        if self.data_table.empty:
            raise RuntimeError("data_table 为空，无法预测。")
        last_time = pd.to_datetime(self.data_table[self.time_col].iloc[-1])
        future = pd.date_range(
            start=last_time + self._time_delta,
            periods=horizon,
            freq=self._time_delta,
        )

        results: Dict[str, np.ndarray] = {}
        ci_bounds: Dict[str, dict] = {}  # model_name → {"lower": ..., "upper": ...}
        for name, inferer in self.vault.models.items():
            try:
                if inferer.paradigm == "ts":
                    last_ds = pd.Timestamp(inferer.model._last_ds)
                    # gap = data_table 末尾到 _last_ds 的步数
                    # 这些步是已观测数据，通过 forecast 额外步数跳过
                    gap = ((last_time.year - last_ds.year) * 12 +
                           (last_time.month - last_ds.month))
                    raw = inferer.forecast(gap + horizon, level=level)
                    pred = np.asarray(raw, dtype=np.float32).ravel()
                    pred = pred[-horizon:]  # 只取真正未来的 horizon 步

                    # 提取 CI
                    ci = getattr(inferer.model, "_last_ci", None)
                    if ci is not None:
                        ci_bounds[name] = {
                            "lower": np.asarray(ci["lower"], dtype=float)[-horizon:],
                            "upper": np.asarray(ci["upper"], dtype=float)[-horizon:],
                        }
                else:
                    lb = inferer.lookback
                    if len(self.data_table) < lb:
                        raise BufferError(
                            f"需要 {lb} 行历史, 当前 {len(self.data_table)}.")
                    # 取 data_table 最后 L 行（末行 = last_time），
                    # predict 输出的 step 0 对应 future[0]（last_time + 1）
                    # 使用 InferencePipeline.predict() 自动处理 transforms
                    x_df = self.data_table.iloc[-lb:][inferer.feature_names]
                    raw = inferer.predict(x_df)  # (N, H, T)，已 inverse-transform
                    raw = np.asarray(raw, dtype=np.float32)
                    n_out = min(horizon, raw.shape[1])
                    pred = raw[0, :n_out, 0]
                    if n_out < horizon:
                        pred = np.pad(pred, (0, horizon - n_out),
                                      'constant', constant_values=np.nan)
                results[name] = pred
            except Exception:
                results[name] = np.full(horizon, np.nan)

        result_df = pd.DataFrame(results, index=future)

        # CI bounds — 追加 _lo / _hi 列
        for mname, bounds in ci_bounds.items():
            result_df[f"{mname}_lo"] = bounds["lower"]
            result_df[f"{mname}_hi"] = bounds["upper"]

        # Risk scoring — attach as additional columns
        if self.risk_scorer is not None:
            for col in result_df.columns:
                risk_col = f"{col}_risk"
                vals = result_df[col].values
                risk_vals = np.array([
                    self.risk_scorer.score(v, t)
                    for v, t in zip(vals, result_df.index)
                ], dtype=int)
                result_df[risk_col] = risk_vals

        result_df.index.name = self.time_col
        return result_df.reset_index()

    # ── Feed ───────────────────────────────────────────────────

    def feed(self, new_data: pd.DataFrame) -> None:
        """追加新观测数据到 data_table。"""
        new_data = new_data.copy()
        self._check_time_continuity(new_data)
        self.data_table = pd.concat(
            [self.data_table, new_data], ignore_index=True)

    # ── TS 模型在线更新 ──────────────────────────────────────

    def update_ts(self, names: Optional[list[str]] = None) -> None:
        """用 data_table 中最近的数据重新拟合 TS 模型。

        保持与原始训练相同的数据量（滑动窗口），只取
        data_table 中最近的 N 行重新拟合，使模型状态
        追赶至最新。不使内部历史无限增长。

        Parameters
        ----------
        names : list of str or None
            要更新的模型名列表。默认更新所有 TS 模型。
        """
        dt = self.data_table
        if dt.empty:
            raise RuntimeError("data_table 为空。")

        for name, inferer in self.vault.models.items():
            if inferer.paradigm != "ts":
                continue
            if names is not None and name not in names:
                continue

            # 读取原始训练数据量：不同模型存储在不同的属性中
            model = inferer.model
            orig_size = (
                getattr(model, "train_size_", None) or       # ARIMA
                getattr(model, "_n_train", None) or           # BSTS
                (len(getattr(model, "_y_train", [])) or None) # ETS/Serfling/STLM/Prophet
            )
            if orig_size is None:
                continue

            # 取 data_table 最近 orig_size 行 → 作为新训练数据
            n = min(orig_size, len(dt))
            dt_slice = dt.iloc[-n:]

            y_raw = dt_slice[inferer.target_names].values.squeeze().astype(float)
            dates = pd.to_datetime(dt_slice[self.time_col]).values

            # 变换到模型期望的空间（data_table 是原始尺度）
            if inferer.transforms is not None:
                y_df = pd.DataFrame(
                    {tn: y_raw for tn in inferer.target_names},
                    index=range(len(y_raw)),
                )
                y_t = inferer.transforms.transform(y_df)
                y_full = y_t[inferer.target_names].values.squeeze().astype(float)
            else:
                y_full = y_raw

            model.fit_sequence(y_full, dates=dates)

            # 更新 _last_ds（trainer 外直接调 fit_sequence 时不自动设置）
            if dates is not None and len(dates) > 0:
                model._last_ds = pd.to_datetime(dates[-1])

    def _find_time_index(self, target_time: pd.Timestamp) -> int:
        """在 data_table 中查找 target_time 的行索引。"""
        times = pd.to_datetime(self.data_table[self.time_col])
        matches = np.where(times == target_time)[0]
        if len(matches) == 0:
            raise KeyError(
                f"时间 {target_time.date()} 不在 data_table 中. "
                f"范围: {times.iloc[0].date()} ~ {times.iloc[-1].date()}")
        return int(matches[-1])

    # ── Time continuity ───────────────────────────────────────

    def _check_time_continuity(self, new_data: pd.DataFrame) -> None:
        new_times = pd.to_datetime(new_data[self.time_col])

        if self.data_table.empty:
            if self._history_end_time is not None:
                expected = self._history_end_time + self._time_delta
                if new_times[0] != expected:
                    self._raise_or_warn(
                        TimeGapError,
                        f"First feed time {new_times[0]} does not follow "
                        f"train_end {self._history_end_time}. Expected {expected}.",
                    )
            elif self.strict:
                self._raise_or_warn(
                    TimeGapError,
                    "First feed: _history_end_time is not set. "
                    "Set runtime._history_end_time = training_data['time'].iloc[-1] "
                    "to enable time continuity checking.",
                )
            return

        last = pd.to_datetime(self.data_table[self.time_col].iloc[-1])
        if new_times[0] <= last:
            self._raise_or_warn(
                TimeOrderError,
                f"Time disorder: last={last}, new={new_times[0]}",
            )

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

    # ── Persistence ───────────────────────────────────────────

    def _save_data_table(self, path: Path) -> None:
        if self.data_table.empty:
            return
        try:
            self.data_table.to_parquet(path / "data_table.parquet")
        except ImportError:
            self.data_table.to_csv(path / "data_table.csv", index=False)

    def save(self, path: Union[str, Path]) -> str:
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        self._path = path
        self._save_data_table(path)
        self.vault.save(str(path / "vault"))
        # Save risk scorer if attached
        if self.risk_scorer is not None:
            with open(path / "risk_scorer.pkl", "wb") as f:
                pickle.dump(self.risk_scorer, f)
        meta = {
            "time_col": self.time_col,
            "time_unit": self.time_unit,
            "strict": self.strict,
            "history_end_time": str(self._history_end_time) if self._history_end_time else None,
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
        runtime._path = path

        if meta["history_end_time"]:
            runtime._history_end_time = pd.Timestamp(meta["history_end_time"])

        dt_path = path / "data_table.parquet"
        if dt_path.exists():
            runtime.data_table = pd.read_parquet(dt_path)
        else:
            csv_path = path / "data_table.csv"
            if csv_path.exists():
                runtime.data_table = pd.read_csv(csv_path)

        # Restore risk scorer if available
        risk_path = path / "risk_scorer.pkl"
        if risk_path.exists():
            with open(risk_path, "rb") as f:
                runtime.risk_scorer = pickle.load(f)

        return runtime

    def __repr__(self) -> str:
        return (
            f"DeploymentRuntime({len(self.vault.models)} models, "
            f"{len(self.data_table)} rows)"
        )

    # ── Refit on full data ────────────────────────────────────────

    def refit(self, verbose: bool = True) -> None:
        """用 ``data_table`` 中的全部数据重新训练所有模型。

        训练完成后，模型内部状态（``_last_ds``、季节性成分等）
        更新至最新时间点。

        Parameters
        ----------
        verbose : bool, default True
            是否打印每个模型的重训状态。
        """
        import inspect

        dt = self.data_table
        if dt.empty:
            raise RuntimeError("data_table 为空，无法重训。")

        for name, inferer in self.vault.models.items():
            if verbose:
                print(f"  refit {name} ...", end=" ", flush=True)
            try:
                # 1. 从当前模型提取构造参数，重建新实例
                model_cls = type(inferer.model)
                sig = inspect.signature(model_cls.__init__)
                init_params = {}
                for pname in list(sig.parameters.keys())[1:]:
                    if hasattr(inferer.model, pname):
                        init_params[pname] = getattr(inferer.model, pname)
                new_model = model_cls(**init_params)

                # 2. 准备全量数据
                y_raw = dt[inferer.target_names].values.squeeze().astype(float)
                dates = pd.to_datetime(dt[self.time_col]).values
                feat_cols = inferer.feature_names or []

                # 3. 变换（data_table 是原始尺度）
                if inferer.transforms is not None:
                    tn = inferer.target_names[0] if len(inferer.target_names) == 1 else inferer.target_names
                    y_df = pd.DataFrame({tn: y_raw} if isinstance(tn, str)
                                        else {t: y_raw[:, i] for i, t in enumerate(tn)})
                    y_t = inferer.transforms.transform(y_df)
                    y_full = y_t.values.squeeze().astype(float)
                else:
                    y_full = y_raw

                # 4. 训练
                if inferer.paradigm == "ts":
                    new_model.fit_sequence(y_full, dates=dates)
                    new_model._last_ds = pd.to_datetime(dates[-1])
                else:
                    from EpiAI.dataset.transforms import build_windows
                    X_raw = dt[feat_cols].values.astype(float) if feat_cols else None
                    win_X, win_y, _ = build_windows(
                        y_full, inferer.lookback, inferer.horizon, X=X_raw,
                    )
                    new_model.fit(win_X, win_y)

                # 5. 替换为新的训练后模型
                inferer.model = new_model
                if verbose:
                    print("OK")
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")

        # 重新拟合风险基线
        if self.risk_scorer is not None:
            try:
                self.risk_scorer.fit(dt)
                if verbose:
                    print("  risk_scorer refit OK")
            except Exception as e:
                if verbose:
                    print(f"  risk_scorer refit FAILED: {e}")

        if verbose:
            print(f"\n全部重训完成，共 {len(self.vault.models)} 个模型。")


__all__ = ["InferencePipeline", "ModelVault", "DeploymentRuntime",
           "TimeGapError", "TimeOrderError", "BufferError"]
