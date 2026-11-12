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


# ── 阈值计算（预警评估共用）─────────────────────────────────

def _compute_threshold(data: pd.DataFrame, target: str,
                       method: str, level: float) -> float:
    """按方法从历史数据计算预警阈值。

    - ``"quantile"``:  level=0.90 → 90% 分位数
    - ``"zscore"``:    level=2.0  → 均值 + 2×标准差
    - ``"threshold"``: level=500  → 绝对值阈值
    """
    vals = data[target].values.astype(float)
    if method == "quantile":
        return float(np.nanquantile(vals, level))
    elif method == "zscore":
        mean = np.nanmean(vals)
        std = np.nanstd(vals) or 1.0
        return float(mean + level * std)
    elif method == "threshold":
        return float(level)
    else:
        raise ValueError(f"不支持的方法: {method}")


def _parse_risk_spec(spec: str) -> tuple[str, str, float]:
    """解析风险策略字符串。

    ``"quantile@0.90"`` → (``"quantile_0_90"``, ``"quantile"``, 0.90)。
    策略名与 ``attach_risk`` 的列名规则一致（``@`` / 小数点 → ``_``）。
    """
    if "@" not in spec:
        raise ValueError(
            f"风险策略需含 @level（如 'quantile@0.90' / 'zscore@1.0' / "
            f"'threshold@500'），收到: {spec!r}")
    method, _, level_str = spec.partition("@")
    if method not in ("quantile", "zscore", "threshold"):
        raise ValueError(
            f"不支持的风险方法: {method!r}（可用 quantile / zscore / threshold）")
    try:
        level = float(level_str)
    except ValueError:
        raise ValueError(
            f"风险策略 level 无法解析为数值: {level_str!r}") from None
    name = spec.replace("@", "_").replace(".", "_")
    return name, method, level


def _safe_ratio(numerator, denominator) -> float:
    """除法，除数为 0 / 任一为 NaN 时返回 NaN。"""
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return num / den


def _youden_operating_point(y_event, y_score) -> dict:
    """扫描模型预测值，求 Youden 最优预警 cutoff（模型侧校准）。

    候选阈值 = 测试段预测值去重升序（与 ROC 各工作点等价）。
    平分时取更低的 cutoff（同 Youden 下更偏向敏感、不漏报）。
    事件缺失或全为事件时无法校准（cutoff 等为 NaN，调用方跳过固化）。

    Returns
    -------
    dict
        ``cutoff, youden, sens, spec, tp, fp, fn, tn, n``
    """
    y_score = np.asarray(y_score, dtype=float).ravel()
    y_event = np.asarray(y_event).ravel()
    mask = ~np.isnan(y_score)
    y_score = y_score[mask]
    y_event = y_event[mask].astype(int)
    n = int(len(y_score))
    n_pos = int((y_event == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        # 无事件 / 全事件 → 无法定义有意义的工作点
        return {"cutoff": np.nan, "youden": np.nan, "sens": np.nan,
                "spec": np.nan, "tp": np.nan, "fp": np.nan,
                "fn": np.nan, "tn": np.nan, "n": n}
    best = None
    for c in np.unique(y_score):
        alert = y_score > c
        tp = int((alert & (y_event == 1)).sum())
        fp = int((alert & (y_event == 0)).sum())
        tn = n_neg - fp
        fn = n_pos - tp
        sens = tp / n_pos
        spec = tn / n_neg
        youden = sens + spec - 1
        if best is None or youden > best[0]:
            best = (youden, float(c), sens, spec, tp, fp, fn, tn)
    if best is None:  # 防御：上面已排除单类情形，正常不会到达
        return {"cutoff": np.nan, "youden": np.nan, "sens": np.nan,
                "spec": np.nan, "tp": np.nan, "fp": np.nan,
                "fn": np.nan, "tn": np.nan, "n": n}
    youden, cutoff, sens, spec, tp, fp, fn, tn = best
    return {"cutoff": cutoff, "youden": youden, "sens": sens, "spec": spec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n}


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
    """自包含的模型单元：评估档案 + 上线模型 + 在线预测，贯穿全生命周期。

    vault 由 ``ForecastTask.compare()`` 创建（评估档案），经 ``fit_full()``
    升级到上线态（运行区 = 全量数据训练的最终权重），此后直接承担
    ``predict()`` / ``feed()`` 等运行职责，是 server 端 load 的唯一单元。

    Directory structure::

        /path/to/vault/
        ├── manifest.json            # all models with metrics + phase
        ├── config.json              # task 档案（问题定义/切分边界）
        ├── transforms.pkl           # 变换定义（供 fit_full 重建）
        ├── data_table.parquet       # 唯一全量原始观测
        ├── compare_records.json     # compare 评估记录（指标表）
        ├── test_predictions/        # 每模型测试集预测（y_true/y_pred）
        ├── fit_history.json         # 历次 fit_full 记录
        └── models/<name>/           # 上线模型（fit_full 权重）
            ├── model.zip
            └── meta.json

    Parameters
    ----------
    models : dict of str → InferencePipeline
        Name → inference pipeline map（运行区：fit_full 后的上线权重）。
    metrics : pd.DataFrame
        Comparison table (model × MAE/RMSE/R²/PearsonR/n) —— compare 档案。
    bundle : PipelineBundle
        最近一次训练/评估使用的数据 bundle。

    Attributes
    ----------
    phase : {"empty", "eval", "full"}
        生命周期阶段：compare 后为 "eval"（只有档案，运行区空，predict 被拒）；
        fit_full 后为 "full"（运行区持全量权重，predict/feed 可用）。
    config : dict
        task 档案：time_col/freq/lookback/horizon/targets/features/transforms/切分边界。
    data_table : pd.DataFrame
        唯一全量原始观测（初始 = task 数据；feed 在此增长）。
    fit_history : list of dict
        历次 fit_full 档案（模型名、时间戳、数据范围、指纹）。
    risk_scorers : dict of str → RiskScorer
        在线预警评分器（attach_risk 挂载，predict 自动附列）。
    risk_cutoffs : dict of str → dict of str → dict
        校准档案：策略名（如 ``quantile_0_90``）→ 模型名 → 校准记录
        {cutoff, youden, sens, spec, event_method, event_level,
        event_threshold, estimated_at}。由 ``evaluate_risk(persist=True)``
        写入；predict 用模型专属 cutoff 做 0/1 报警判定（feed/重训不动）。
    test_predictions : dict of str → pd.DataFrame
        compare 档案：每模型 {time, y_true, y_pred}。
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

        # ── 生命周期 / 档案 ──────────────────────────────────
        self.phase: str = "empty"
        self.config: dict = {}
        self._transforms_def = None   # 变换定义（未随训练 fit 的原始配置）
        self.fit_history: list[dict] = []
        self.test_predictions: dict[str, pd.DataFrame] = {}
        self.metrics_origin: str | None = None
        # 最近一次建档的训练失败明细（task.compare / task.fit_full 填充，瞬态不持久化）
        self.compare_failures: Optional[list[dict]] = None
        self.fit_failures: Optional[list[dict]] = None

        # ── 在线运行态 ───────────────────────────────────────
        self.data_table: pd.DataFrame = pd.DataFrame()
        self.risk_scorers: dict[str, object] = {}
        self.risk_cutoffs: dict[str, dict[str, dict]] = {}
        self.strict: bool = True
        self._path: Optional[Path] = None
        self._time_delta = None
        self._history_end_time: Optional[pd.Timestamp] = None

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

            m = result.metrics.iloc[0] if len(result.metrics) else None
            if m is not None:
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
            else:
                # 全量训练模型（fit_full）无留出指标 → 占位 NaN 行
                rows.append({
                    "model": name,
                    "paradigm": result.model.paradigm(),
                    "MAE": float("nan"),
                    "RMSE": float("nan"),
                    "MAPE": float("nan"),
                    "R2": float("nan"),
                    "PearsonR": float("nan"),
                    "n": float("nan"),
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

    # ════════════════════════════════════════════════════════════════
    # 生命周期档案（由 ForecastTask 建档时调用）
    # ════════════════════════════════════════════════════════════════

    def set_task_archive(self, task, data_df: pd.DataFrame) -> "ModelVault":
        """写入 task 档案（问题定义 + 切分边界）并接管全量原始数据。

        由 ``ForecastTask.compare()`` / ``fit_full()`` 建档时调用。
        之后 vault 自带重建所需全部配置，可脱离 task 独立 fit_full。
        """
        self.config = {
            "time_col": task.time_col,
            "freq": task.freq,
            "lookback": task.lookback,
            "horizon": task.horizon,
            "targets": list(task.target),
            "features": list(task.features) if task.features else None,
            "split": {
                "train_end": str(task.split.train_end) if task.split.train_end else None,
                "val_end": str(task.split.val_end) if task.split.val_end else None,
                "train_ratio": task.split.train_ratio,
                "val_ratio": task.split.val_ratio,
            },
            "created": str(pd.Timestamp.now()),
        }
        from copy import deepcopy
        self._transforms_def = deepcopy(task.transforms)
        self.data_table = data_df.copy().reset_index(drop=True)
        if self.config["time_col"] in self.data_table.columns:
            self.data_table[self.config["time_col"]] = pd.to_datetime(
                self.data_table[self.config["time_col"]]
            )
            self.data_table = self.data_table.sort_values(
                self.config["time_col"]).reset_index(drop=True)
        self._sync_time_state()
        return self

    def _sync_time_state(self) -> None:
        """从 config/data_table 同步时间参数。"""
        freq = self.config.get("freq") or "MS"
        self._time_delta = pd.tseries.frequencies.to_offset(freq)
        if not self.data_table.empty:
            tc = self.config.get("time_col")
            if tc and tc in self.data_table.columns:
                self._history_end_time = pd.to_datetime(
                    self.data_table[tc].iloc[-1])

    # ── 阶段守卫 ────────────────────────────────────────────────

    def _require_online(self, action: str = "predict") -> None:
        """predict/feed 等运行操作要求 vault 已进入上线态。"""
        if self.phase not in ("full", "legacy"):
            raise ValueError(
                f"vault 处于 phase={self.phase!r}，无法执行 {action}()。"
                "请先调用 vault.fit_full([...]) 用全量数据训练上线模型。"
                "流程: vault = task.compare([...]) → vault.fit_full([best]) → vault.predict(...)"
            )

    # ════════════════════════════════════════════════════════════════
    # 在线预测 / 数据接入
    # ════════════════════════════════════════════════════════════════

    def predict(self, horizon: int = 1,
                level: Optional[int] = None) -> pd.DataFrame:
        """预测未来 horizon 步，返回 时间×模型 表格（原始尺度）。

        要求 vault 已 ``fit_full()``（运行区为全量训练的最终权重），
        因此 TS 模型的 ``_last_ds`` 与 data_table 末尾对齐，直接从
        最新观测外推（gap=0）。

        Parameters
        ----------
        horizon : int
            预测步数，默认 1。
        level : int or None, optional
            置信水平（如 95）。支持此参数的模型会在原预测值旁追加
            ``{model}_lo`` / ``{model}_hi`` 列。

        Returns
        -------
        pd.DataFrame
            行 = 未来时间点，列 = 模型名（及 CI / 风险等级列）。
        """
        self._require_online("predict")
        if self.data_table.empty:
            raise RuntimeError("data_table 为空，无法预测。")
        tc = self.config.get("time_col", "time")
        last_time = pd.to_datetime(self.data_table[tc].iloc[-1])
        future = pd.date_range(
            start=last_time + self._time_delta,
            periods=horizon,
            freq=self._time_delta,
        )

        results: Dict[str, np.ndarray] = {}
        ci_bounds: Dict[str, dict] = {}
        for name, inferer in self.models.items():
            try:
                if inferer.paradigm == "ts":
                    last_ds = pd.Timestamp(inferer.model._last_ds)
                    # gap = data_table 末尾到 _last_ds 的步数（fit_full 后为 0），
                    # 按 vault 自身 freq 计步，月/周/日通用
                    gap = len(pd.date_range(last_ds, last_time,
                                            freq=self._time_delta)) - 1
                    raw = inferer.forecast(gap + horizon, level=level)
                    pred = np.asarray(raw, dtype=np.float32).ravel()
                    pred = pred[-horizon:]
                    ci = getattr(inferer.model, "_last_ci", None)
                    if ci is not None:
                        ci_bounds[name] = {
                            "lower": np.asarray(ci["lower"], dtype=float)[-horizon:],
                            "upper": np.asarray(ci["upper"], dtype=float)[-horizon:],
                        }
                else:
                    lb = inferer.lookback
                    if len(self.data_table) < lb:
                        raise RuntimeError(
                            f"需要 {lb} 行历史, 当前 {len(self.data_table)}.")
                    x_df = self.data_table.iloc[-lb:][inferer.feature_names]
                    raw = inferer.predict(x_df)
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
        for mname, bounds in ci_bounds.items():
            result_df[f"{mname}_lo"] = bounds["lower"]
            result_df[f"{mname}_hi"] = bounds["upper"]

        # 在线风险列：模型专属校准 cutoff（evaluate_risk persist）优先，
        # 无校准档案时回退到 RiskScorer 统计阈值判定（旧流程兜底）。
        model_cols = set(results.keys())
        strategy_names = sorted(set(self.risk_scorers) | set(self.risk_cutoffs))
        for sname in strategy_names:
            suffix = f"_{sname}" if sname != "risk" else "_risk"
            cutoffs_tbl = self.risk_cutoffs.get(sname) or {}
            scorer = self.risk_scorers.get(sname)
            for col in model_cols:
                risk_col = f"{col}{suffix}"
                vals = result_df[col].values
                if cutoffs_tbl:
                    # 校准模式：预测值 > 该模型专属 cutoff → 报警(1)。
                    # 预测失败(NaN)或该模型未校准 → NaN（明确不给判定，
                    # 不静默套用统计阈值）。
                    rec = cutoffs_tbl.get(col)
                    if rec is not None and np.isfinite(float(rec["cutoff"])):
                        c = float(rec["cutoff"])
                        result_df[risk_col] = np.where(
                            np.isnan(vals), np.nan,
                            (np.asarray(vals, dtype=float) > c).astype(float))
                    else:
                        result_df[risk_col] = np.nan
                elif scorer is not None:
                    result_df[risk_col] = np.array([
                        scorer.score(float(v), t)
                        for v, t in zip(vals, result_df.index)
                    ], dtype=int)

        result_df.index.name = tc
        return result_df.reset_index()

    def feed(self, new_data: pd.DataFrame) -> None:
        """追加新观测到 data_table（模型冻结，不触发重训）。

        新数据首条记录须严格接续 data_table 末条记录（按 freq 步进）；
        时间断裂时抛 ``TimeGapError``（可设 ``vault.strict = False`` 降级为警告）。
        """
        new_data = new_data.copy()
        tc = self.config.get("time_col", "time")
        if tc in new_data.columns:
            new_data[tc] = pd.to_datetime(new_data[tc])
        self._check_time_continuity(new_data)
        self.data_table = pd.concat(
            [self.data_table, new_data], ignore_index=True)
        self._sync_time_state()

    def update_ts(self, names: Optional[list[str]] = None) -> None:
        """用 data_table 最近的数据重新拟合 TS 模型（滑窗追平）。

        常规更新用 ``feed()`` + 预测即可；结构性变化后应走
        ``vault.fit_full()`` 全量重训。此方法保留用于兼容场景。
        """
        self._require_online("update_ts")
        dt = self.data_table
        if dt.empty:
            raise RuntimeError("data_table 为空。")
        tc = self.config.get("time_col", "time")

        for name, inferer in self.models.items():
            if inferer.paradigm != "ts":
                continue
            if names is not None and name not in names:
                continue
            model = inferer.model
            orig_size = (
                getattr(model, "train_size_", None) or
                getattr(model, "_n_train", None) or
                (len(getattr(model, "_y_train", [])) or None)
            )
            if orig_size is None:
                continue
            n = min(orig_size, len(dt))
            dt_slice = dt.iloc[-n:]
            y_raw = dt_slice[inferer.target_names].values.squeeze().astype(float)
            dates = pd.to_datetime(dt_slice[tc]).values
            if inferer.transforms is not None:
                y_df = pd.DataFrame(
                    {tn: y_raw for tn in inferer.target_names},
                    index=range(len(y_raw)),
                )
                y_full = inferer.transforms.transform(y_df)[
                    inferer.target_names].values.squeeze().astype(float)
            else:
                y_full = y_raw
            model.fit_sequence(y_full, dates=dates)
            if dates is not None and len(dates) > 0:
                model._last_ds = pd.to_datetime(dates[-1])

    # ── 时间连续性校验（自 data_table）────────────────────────

    def _check_time_continuity(self, new_data: pd.DataFrame) -> None:
        new_times = pd.to_datetime(new_data[self.config.get("time_col", "time")])
        if self.data_table.empty:
            if self._history_end_time is not None:
                expected = self._history_end_time + self._time_delta
                if new_times[0] != expected:
                    self._raise_or_warn(
                        TimeGapError,
                        f"First feed time {new_times[0]} does not follow "
                        f"train_end {self._history_end_time}. Expected {expected}.",
                    )
            return
        last = pd.to_datetime(self.data_table[self.config.get("time_col", "time")].iloc[-1])
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
                    f"Time gap: last={last}, expected={expected_i}, got={t}.",
                )

    def _raise_or_warn(self, exc_type, msg: str) -> None:
        if self.strict:
            raise exc_type(msg)
        import warnings
        warnings.warn(f"[ModelVault] {msg}", stacklevel=3)

    # ════════════════════════════════════════════════════════════════
    # 模型更新：fit_full（全量数据训练，升级到上线态）
    # ════════════════════════════════════════════════════════════════

    def fit_full(self,
                 model_names: Optional[list[str]] = None,
                 cache_dir: Optional[str | Path] = None,
                 verbose: bool = True,
                 risk=None) -> "ModelVault":
        """用 data_table 的 100% 数据训练最终上线模型（无留出）。

        vault 自带 task 档案与全量数据，因此可脱离 ForecastTask 独立执行
        （server 端 load 后可直接调用）。升级过程:

        - 运行区 ``models`` 替换为全量训练权重
        - ``phase`` → ``"full"``，此后 ``predict()`` / ``feed()`` 可用
        - ``fit_history`` 追加一条档案（时间戳/数据范围），可追溯
        - 原 compare 评估档案（metrics/test_predictions）保持不变

        Parameters
        ----------
        model_names : list of str or None
            要全量训练的模型名。默认 ``None`` 时使用当前运行区模型
            （即上一次 fit_full 的模型集）。
        cache_dir : str or Path or None, optional
            透传给内部训练流程的缓存目录。
        verbose : bool, default True
        risk : optional
            若提供，等价于随后调用 ``attach_risk(risk)``。

        Returns
        -------
        ModelVault
            self（原地升级）。
        """
        if not self.config:
            raise ValueError("vault 缺少 task 档案（config 为空），无法重建训练流程。")

        names = model_names or list(self.models.keys())
        if not names:
            raise ValueError(
                "未指定要训练的模型：vault 运行区为空且未传 model_names。"
                "请显式传入: vault.fit_full([\"sf_auto_ets\", ...])")

        from copy import deepcopy

        from EpiAI import ForecastTask
        from EpiAI.dataset import TimeSplit

        transforms = deepcopy(self._transforms_def) if self._transforms_def is not None else None
        task = ForecastTask(
            data=self.data_table,
            target=self.config["targets"],
            features=self.config.get("features"),
            time_col=self.config.get("time_col", "time"),
            freq=self.config.get("freq"),
            horizon=self.config.get("horizon", 3),
            lookback=self.config.get("lookback", 12),
            transforms=transforms,
            split=TimeSplit(train_ratio=1.0),
        )

        full_vault = task.fit_full(list(dict.fromkeys(names)),
                                   cache_dir=cache_dir,
                                   eval_vault=None,
                                   verbose=verbose)
        # 运行区替换为全量权重
        self.models = full_vault.models
        self.bundle = full_vault.bundle
        # 无 compare 档案时补 metrics（NaN 占位），有则保留评估档案
        if self.metrics.empty:
            self.metrics = full_vault.metrics
        # 透传单模型失败明细（task.fit_full 的 skip_on_error 记录）
        self.fit_failures = full_vault.fit_failures

        self.fit_history.append({
            "models": list(full_vault.models.keys()),
            "time": str(pd.Timestamp.now()),
            "n_rows": len(self.data_table),
            "last_time": str(self._history_end_time) if self._history_end_time else None,
            "phase_prev": self.phase,
        })
        self.phase = "full"
        self._sync_time_state()

        if risk is not None:
            self.attach_risk(risk)
        return self

    # ════════════════════════════════════════════════════════════════
    # 风险：在线评分器挂载 + 离线档案评估（AUC 等）
    # ════════════════════════════════════════════════════════════════

    def attach_risk(self, risk) -> "ModelVault":
        """挂载在线风险评分器（基于百分比/绝对值等的区间划分）。

        支持与 ``ForecastTask.deploy(risk=...)`` 相同的语法::

            vault.attach_risk("quantile")          # 多级 0-3（默认 70/90/95%）
            vault.attach_risk("quantile@0.80")     # 二分类预警
            vault.attach_risk(["quantile", "zscore@1.0", "threshold@500"])

        评分器基线 fit 在当前 data_table 上；之后每次 ``predict()``
        自动附加 ``{model}_{name}`` 风险列。
        """
        from EpiAI.risk import RiskScorer

        risk_list: list[tuple[str, str | RiskScorer]] = []
        items = risk if isinstance(risk, list) else [risk]
        tc = self.config.get("time_col", "time")
        value_col = (self.config.get("targets") or [None])[0]

        for item in items:
            if isinstance(item, str):
                if "@" in item:
                    method, level_str = item.split("@", 1)
                    # 列名去掉小数点：quantile@0.90 → quantile_0_90
                    name = item.replace("@", "_").replace(".", "_")
                    risk_list.append((name, (method, float(level_str))))
                else:
                    risk_list.append((item, item))
            elif isinstance(item, RiskScorer):
                name = getattr(item, "method", "risk")
                risk_list.append((name, item))
            else:
                raise TypeError(f"Unsupported risk type: {type(item)}")

        for name, spec in risk_list:
            if isinstance(spec, str):
                scorer = RiskScorer(method=spec, time_col=tc, value_col=value_col)
            elif isinstance(spec, tuple):
                method, level = spec
                bounds_map = {
                    "quantile": [level, level, level],
                    "zscore": [level, level, level],
                    "threshold": [level, level, level],
                }
                scorer = RiskScorer(
                    method=method,
                    quantile_bounds=bounds_map["quantile"],
                    zscore_bounds=bounds_map["zscore"],
                    thresholds=bounds_map["threshold"],
                    binary=True,
                    time_col=tc,
                    value_col=value_col,
                )
            else:
                scorer = spec
            scorer.fit(self.data_table)
            self.risk_scorers[name] = scorer
        return self

    def evaluate_risk(self,
                      risk: Optional[Union[str, list[str]]] = None,
                      method: str = "quantile",
                      level: float = 0.90,
                      event_method: Optional[str] = None,
                      event_level: Optional[float] = None,
                      persist: bool = False,
                      verbose: bool = False,
                      ) -> pd.DataFrame:
        """校准式预警评估：为每个 (模型 × 风险策略) 估计 Youden 最优 cutoff。

        分两层语义（事件规则与模型阈值解耦）：

        ① 事件定义（业务侧，与模型无关）—— 风险策略
        （``quantile@0.90`` / ``zscore@1.0`` / ``threshold@500`` …）作用于
        **测试段 y_true**，二值化为事件 0/1；基准统计（分位/均值±σ）只取
        训练段（time < train_end，见 ``_eval_threshold_history``），与后续
        feed 解耦、可复现。

        ② 模型侧校准 —— 对每个模型在测试段 y_pred 上扫描候选阈值，取
        Youden（sens + spec − 1）最大的点作为该模型在该策略下的专属
        ``cutoff``。上线判定即 ``预测值 > cutoff``，取代旧版"所有模型
        共用一个统计阈值"的判定。

        Parameters
        ----------
        risk : str or list of str, optional
            一个或多个风险策略（``@level`` 语法，如 ``\"quantile@0.90\"``）。
            缺省时用 ``method``/``level`` 拼成单个策略（兼容旧签名）。
        method : str, default \"quantile\"
            risk=None 时的事件方法（quantile / zscore / threshold）。
        level : float, default 0.90
            risk=None 时的事件级别。
        event_method, event_level : str/float or None
            可选：显式覆盖事件定义（默认沿用策略本身的方法/级别）。
        persist : bool, default False
            是否将本次校准结果固化进 ``vault.risk_cutoffs``。只有
            persist=True 才写入/更新；feed()、predict()、模型重训均不
            改变已固化的 cutoff —— 换策略 = 换 spec 重新 persist。
        verbose : bool, default False
            persist 时打印固化摘要。

        Returns
        -------
        pd.DataFrame
            逐 (strategy, model) 一行：
            ``strategy, model, auc, cutoff, youden, sens, spec, hit_rate,
            fa_rate, precision, recall, f1, n_alerts, n_events, tp, fp,
            fn, tn``。cutoff 为 NaN 表示该模型无法校准（测试段无事件 /
            全事件 / 预测全 NaN），不会写入 risk_cutoffs。
        """
        if not self.test_predictions:
            raise ValueError(
                "vault 无 compare 档案（test_predictions 为空）。"
                "请先执行 task.compare() 建档。")

        history = self._eval_threshold_history()
        target = (self.config.get("targets") or [None])[0]
        if target is None or history is None or history.empty:
            raise ValueError(
                "vault 档案缺少训练数据/目标列，无法计算事件基准。")
        assert history is not None

        if risk is None:
            specs = [f"{method}@{level}"]
        elif isinstance(risk, str):
            specs = [risk]
        else:
            specs = list(risk)

        records: list[dict] = []
        for spec in specs:
            sname, smethod, slevel = _parse_risk_spec(spec)
            em = event_method or smethod
            el = float(event_level) if event_level is not None else slevel
            event_threshold = _compute_threshold(history, target, em, el)

            cutoffs_tbl: dict[str, dict] = {}
            for model_name, td in self.test_predictions.items():
                y_true = td["y_true"].values.astype(float)
                y_pred = td["y_pred"].values.astype(float)
                events = (y_true > event_threshold).astype(int)
                op = _youden_operating_point(events, y_pred)

                auc = np.nan
                try:
                    from sklearn.metrics import roc_auc_score
                    valid = ~np.isnan(y_pred)
                    auc = float(roc_auc_score(events[valid], y_pred[valid]))
                except Exception:
                    auc = np.nan

                recall = op["sens"]                       # Youden 点上的召回
                precision = _safe_ratio(op["tp"], op["tp"] + op["fp"])
                n_alerts = op["tp"] + op["fp"] if np.isfinite(op["tp"]) else np.nan
                records.append({
                    "strategy": sname,
                    "model": model_name,
                    "auc": auc,
                    "cutoff": op["cutoff"],
                    "youden": op["youden"],
                    "sens": op["sens"],
                    "spec": op["spec"],
                    "hit_rate": recall,
                    "fa_rate": _safe_ratio(op["fp"], op["fp"] + op["tn"]),
                    "precision": precision,
                    "recall": recall,
                    "f1": _safe_ratio(2 * precision * recall, precision + recall),
                    "n_alerts": n_alerts,
                    "n_events": int((events == 1).sum()),
                    "tp": op["tp"], "fp": op["fp"],
                    "fn": op["fn"], "tn": op["tn"],
                })

                if persist and np.isfinite(op["cutoff"]):
                    cutoffs_tbl[model_name] = {
                        "cutoff": float(op["cutoff"]),
                        "youden": float(op["youden"]),
                        "sens": float(op["sens"]),
                        "spec": float(op["spec"]),
                        "auc": auc if np.isfinite(auc) else None,
                        "event_method": em,
                        "event_level": float(el),
                        "event_threshold": float(event_threshold),
                        "estimated_at": str(pd.Timestamp.now()),
                    }
            if persist and cutoffs_tbl:
                # 整表写回：同策略重估以本次为准（含被剔除的模型）
                self.risk_cutoffs[sname] = cutoffs_tbl
                if verbose:
                    print(f"[evaluate_risk] 固化 {len(cutoffs_tbl)} 个模型 cutoff "
                          f"→ risk_cutoffs['{sname}'] (event_threshold="
                          f"{event_threshold:.4g})")

        if not records:
            raise RuntimeError("无评估记录生成。")
        cols = ["strategy", "model", "auc", "cutoff", "youden", "sens", "spec",
                "hit_rate", "fa_rate", "precision", "recall", "f1",
                "n_alerts", "n_events", "tp", "fp", "fn", "tn"]
        df = pd.DataFrame(records)[cols]
        # 输出顺序：策略按传入顺序分组，组内按 auc 降序（NaN 排最后）。
        # lexsort 以最后一个键为主排序键。
        order_map = {s: i for i, s in enumerate(df["strategy"].unique())}
        strat_codes = np.array([order_map[s] for s in df["strategy"]])
        auc_arr = df["auc"].to_numpy(dtype=float)
        auc_key = np.where(np.isnan(auc_arr), np.inf, -auc_arr)  # NaN 排最后
        return df.iloc[np.lexsort((auc_key, strat_codes))].reset_index(drop=True)

    def _eval_threshold_history(self):
        """返回阈值基准：data_table 训练段（time < train_end）的 DataFrame。"""
        if self.data_table.empty:
            return None
        tc = self.config.get("time_col", "time")
        train_end = (self.config.get("split") or {}).get("train_end")
        if train_end:
            return self.data_table[
                pd.to_datetime(self.data_table[tc]) < pd.Timestamp(train_end)]
        return self.data_table

    def save(self, path: Union[str, Path]) -> str:
        """持久化 vault（档案 + 上线模型 + 全量数据）到目录。

        Directory::

            manifest.json        # models/metrics/phase
            config.json          # task 档案（问题定义 + 切分边界）
            transforms.pkl       # 变换定义（供 fit_full 重建）
            data_table.parquet   # 唯一全量原始观测
            test_predictions/    # compare 档案：每模型测试集预测
            fit_history.json     # 历次 fit_full 记录
            risk_*.pkl           # 在线风险评分器（若有）
            risk_cutoffs.json    # 校准档案（模型专属 cutoff，若有）
            <name>/model.zip     # 上线模型包

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
        self._path = path

        # ── 运行区模型 ────────────────────────────────────────
        for name, pip in self.models.items():
            model_dir = path / name
            model_dir.mkdir(exist_ok=True)
            pip.save(str(model_dir / "model.zip"))
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

        # ── 档案 / 数据 / 配置 ────────────────────────────────
        manifest = {
            "models": list(self.models.keys()),
            "n_models": len(self.models),
            "phase": self.phase,
            "created": str(pd.Timestamp.now()),
        }
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
        if self.config:
            (path / "config.json").write_text(
                json.dumps(self.config, indent=2, ensure_ascii=False))
        (path / "transforms.pkl").write_bytes(
            pickle.dumps(self._transforms_def) if self._transforms_def is not None else b"")
        (path / "fit_history.json").write_text(
            json.dumps(self.fit_history, indent=2, ensure_ascii=False))
        if not self.data_table.empty:
            try:
                self.data_table.to_parquet(path / "data_table.parquet")
            except ImportError:
                self.data_table.to_csv(path / "data_table.csv", index=False)
        # compare 档案：测试集预测
        tp_dir = path / "test_predictions"
        tp_dir.mkdir(exist_ok=True)
        for name, td in self.test_predictions.items():
            td.to_csv(tp_dir / f"{name}.csv", index=False)
        # 在线风险评分器
        for sname, scorer in self.risk_scorers.items():
            with open(path / f"risk_{sname}.pkl", "wb") as f:
                pickle.dump(scorer, f)
        # 校准档案（模型专属 cutoff，evaluate_risk(persist=True) 固化）
        if self.risk_cutoffs:
            (path / "risk_cutoffs.json").write_text(
                json.dumps(self.risk_cutoffs, indent=2, ensure_ascii=False))
        return str(path.resolve())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelVault":
        """加载 vault 目录（server 端唯一入口）。

        支持两种格式：
        - 新格式：含 config.json / data_table.parquet 的自包含 vault
        - 旧格式（DeploymentRuntime 目录或纯 vault 目录）：无 config 时
          视作 legacy，尽量从 manifest/模型恢复；phase 标记为 "legacy"。
        """
        path = Path(path)
        if not (path / "manifest.json").exists():
            # 旧 DeploymentRuntime 目录（顶层 runtime_meta.json + vault/）
            if (path / "runtime_meta.json").exists():
                return DeploymentRuntime.load(path)
            raise FileNotFoundError(f"Not a vault directory: {path}")
        manifest = json.loads((path / "manifest.json").read_text())
        metrics = pd.DataFrame(manifest["metrics"]).T
        metrics.index.name = "model"

        models: Dict[str, InferencePipeline] = {}
        for name in manifest["models"]:
            model_zip = path / name / "model.zip"
            if model_zip.exists():
                models[name] = InferencePipeline.load(str(model_zip))

        vault = cls(models, metrics, bundle=None)
        vault.phase = manifest.get("phase", "legacy")
        vault._path = path

        # 兼容旧 runtime 目录：模型在 vault/ 子目录
        if not models and (path / "vault").exists():
            return DeploymentRuntime.load(path)

        if (path / "config.json").exists():
            vault.config = json.loads((path / "config.json").read_text())
        elif models:
            # legacy：从任一模型推断最小配置
            ref = next(iter(models.values()))
            vault.config = {
                "time_col": "time",
                "freq": "MS",
                "lookback": ref.lookback,
                "horizon": ref.horizon,
                "targets": ref.target_names,
                "features": ref.feature_names,
                "split": {},
                "created": None,
            }
            vault.phase = "legacy"

        t_path = path / "transforms.pkl"
        if t_path.exists():
            raw = t_path.read_bytes()
            vault._transforms_def = pickle.loads(raw) if raw else None

        # 全量数据
        dt_path = path / "data_table.parquet"
        if dt_path.exists():
            vault.data_table = pd.read_parquet(dt_path)
        else:
            csv_path = path / "data_table.csv"
            if csv_path.exists():
                vault.data_table = pd.read_csv(csv_path)

        # compare 档案：测试集预测
        tp_dir = path / "test_predictions"
        if tp_dir.exists():
            for csv in sorted(tp_dir.glob("*.csv")):
                vault.test_predictions[csv.stem] = pd.read_csv(csv)

        # fit 历史 / 风险评分器
        fh = path / "fit_history.json"
        if fh.exists():
            vault.fit_history = json.loads(fh.read_text())
        for pkl in sorted(path.glob("risk_*.pkl")):
            sname = pkl.stem.replace("risk_", "", 1)
            with open(pkl, "rb") as f:
                vault.risk_scorers[sname] = pickle.load(f)

        # 校准档案（模型专属 cutoff）
        rc_path = path / "risk_cutoffs.json"
        if rc_path.exists():
            vault.risk_cutoffs = json.loads(rc_path.read_text())

        vault._sync_time_state()
        return vault

    def __repr__(self) -> str:
        return (f"ModelVault(phase={self.phase}, {len(self.models)} models, "
                f"{len(self.data_table)} rows)")


# =====================================================================
# DeploymentRuntime — production deployment with unified data table
# =====================================================================

class TimeGapError(ValueError):
    """Raised when new data creates a time gap."""

class TimeOrderError(ValueError):
    """Raised when new data is older than the latest record."""

class BufferError(RuntimeError):
    """Raised when data_table has fewer rows than a model's lookback."""


class DeploymentRuntime(ModelVault):
    """[兼容壳] 旧「部署运行时」—— v0.8 起职责已并入 ModelVault。

    ModelVault 现为唯一生命周期单元：自带 task 档案、全量数据表与
    在线 predict()/feed()/fit_full()/evaluate_risk() 能力。

    本类仅为旧代码/旧目录的平滑迁移保留：构造签名、refit()、save/load
    沿用旧用法但内部委托 ModelVault。新代码请直接使用 ModelVault。
    """

    def __init__(self,
                 vault: Optional["ModelVault"] = None,
                 time_col: str = "time",
                 time_unit: str = "MS",
                 strict: bool = True,
                 data_table: Optional[pd.DataFrame] = None):
        if vault is None:
            vault = ModelVault({}, pd.DataFrame(), None)
        super().__init__(models=vault.models, metrics=vault.metrics,
                         bundle=vault.bundle)
        # 状态迁移
        self.phase = getattr(vault, "phase", "full")
        self.config = dict(getattr(vault, "config", {}) or {})
        self._transforms_def = getattr(vault, "_transforms_def", None)
        self.fit_history = list(getattr(vault, "fit_history", []))
        self.test_predictions = dict(getattr(vault, "test_predictions", {}))
        self.risk_scorers = dict(getattr(vault, "risk_scorers", {}))
        self.risk_cutoffs = dict(getattr(vault, "risk_cutoffs", {}))
        self.strict = strict
        if not self.config:
            self.config = {
                "time_col": time_col, "freq": time_unit,
                "lookback": 12, "horizon": 3,
                "targets": [], "features": None, "split": {},
                "created": None,
            }
        self.config.setdefault("time_col", time_col)
        self.config.setdefault("freq", time_unit)
        if data_table is not None:
            self.data_table = data_table.copy()
        else:
            self.data_table = getattr(vault, "data_table", pd.DataFrame()).copy()
        tc = self.config.get("time_col", "time")
        if not self.data_table.empty and tc in self.data_table.columns:
            self.data_table[tc] = pd.to_datetime(self.data_table[tc])
        self._sync_time_state()

    # ── 兼容：refit → fit_full ─────────────────────────────────

    def refit(self, model_names: Optional[list[str]] = None,
              cache_dir: Optional[str | Path] = None,
              verbose: bool = True):
        """[弃用] 等价于 ModelVault.fit_full()（全量数据重训上线模型）。"""
        import warnings
        warnings.warn(
            "DeploymentRuntime.refit() 已由 ModelVault.fit_full() 取代："
            "vault 自带全量数据，直接 vault.fit_full(...) 即可。",
            DeprecationWarning, stacklevel=2)
        return self.fit_full(model_names=model_names, cache_dir=cache_dir,
                             verbose=verbose)

    # ── 兼容：save/load（新格式与 ModelVault 互通）───────────

    def save(self, path: Union[str, Path]) -> str:
        """持久化（输出新 vault 格式，ModelVault.load 可直接读取）。"""
        return ModelVault.save(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeploymentRuntime":
        """加载部署目录；兼容旧 runtime_meta.json 格式与 ModelVault.load。"""
        path = Path(path)
        if (path / "config.json").exists() or not (path / "runtime_meta.json").exists():
            return ModelVault.load(path)
        # ── 旧 DeploymentRuntime 目录 ──────────────────────────
        meta = json.loads((path / "runtime_meta.json").read_text())
        vault_path = path / "vault"
        vault = (ModelVault.load(vault_path) if vault_path.exists()
                 else ModelVault({}, pd.DataFrame(), None))
        runtime = cls(vault=vault,
                      time_col=meta.get("time_col", "time"),
                      time_unit=meta.get("time_unit", "MS"),
                      strict=meta.get("strict", True))
        dt_path = path / "data_table.parquet"
        if dt_path.exists():
            runtime.data_table = pd.read_parquet(dt_path)
        else:
            csv_path = path / "data_table.csv"
            if csv_path.exists():
                runtime.data_table = pd.read_csv(csv_path)
        tc = runtime.config.get("time_col", "time")
        if not runtime.data_table.empty and tc in runtime.data_table.columns:
            runtime.data_table[tc] = pd.to_datetime(runtime.data_table[tc])
        if meta.get("history_end_time"):
            runtime._history_end_time = pd.Timestamp(meta["history_end_time"])
        for pkl in sorted(path.glob("risk_*.pkl")):
            sname = pkl.stem.replace("risk_", "", 1)
            with open(pkl, "rb") as f:
                runtime.risk_scorers[sname] = pickle.load(f)
        runtime.phase = "legacy"
        runtime._sync_time_state()
        return runtime


__all__ = ["InferencePipeline", "ModelVault", "DeploymentRuntime",
           "TimeGapError", "TimeOrderError", "BufferError"]
