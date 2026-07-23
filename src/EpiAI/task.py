"""
ForecastTask — one-shot forecast → compare → deploy.

Usage::

    task = ForecastTask(
        data=df,
        target="cases",
        features=["t2m"],
        time_col="time",
        freq="MS",
        horizon=6,
        split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
        transforms=Compose([StandardScaler()]),
    )

    # Train multiple models and compare
    vault = task.compare(["sf_auto_ets", "sf_holt_winters", "xgb"])

    # Deploy with risk scoring
    runtime = task.deploy(vault.best("PearsonR"), vault,
                          risk=["quantile", "zscore"])

    # Production use
    report = runtime.predict(horizon=6, level=95)
    # → columns: model_pred | model_lo | model_hi | model_quantile_risk | model_zscore_risk
    runtime.feed(new_df)
    runtime.refit(model_names=["sf_auto_ets"])  # 指定重训某个模型
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, SlidingWindow,
)
from EpiAI.dataset.pipeline import PipelineBundle
from EpiAI.inference import DeploymentRuntime, ModelVault
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer


# ── Frequency helpers ─────────────────────────────────────────────

_SEASON_LENGTH_MAP = {
    "MS": 12,    # month start
    "M": 12,     # month end
    "D": 7,      # day
    "Q": 4,      # quarter
    "QS": 4,     # quarter start
    "Y": 1,      # year
    "YS": 1,     # year start
    "W": 52,     # week
    "h": 24,     # hour
}


def infer_freq(series: pd.Series) -> str | None:
    """Infer frequency from a datetime series, falling back to median gap."""
    from pandas.tseries.frequencies import to_offset

    freq = pd.infer_freq(series.iloc[:10])
    if freq is not None:
        return freq

    # Fall back to median gap
    diffs = pd.to_datetime(series).diff().dropna()
    if len(diffs) == 0:
        return None
    median = diffs.median()
    try:
        return to_offset(median).freqstr
    except Exception:
        return None


def freq_to_season_length(freq: str | None) -> int:
    """Convert a pandas frequency string to a seasonal period."""
    if freq is None:
        return 1
    # Strip multiplier prefix (e.g. "2MS" → "MS")
    base = "".join(c for c in freq if not c.isdigit())
    return _SEASON_LENGTH_MAP.get(base, 12)


def freq_to_time_unit(freq: str | None) -> str:
    """Convert a pandas frequency string to a DeploymentRuntime time_unit."""
    if freq is None:
        return "MS"
    base = "".join(c for c in freq if not c.isdigit())
    return base if base else "MS"


# ── ForecastTask ──────────────────────────────────────────────────

class ForecastTask:
    """Define a forecasting problem and run the full workflow.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to CSV, or a DataFrame.
    target : str or list of str
        Target column name(s).
    features : list of str or None, optional
        Feature column names.  When ``None``, only the target's own
        lags are used (autoregressive).
    time_col : str, default ``"time"``
    freq : str or None, optional
        Pandas frequency string (``"MS"``, ``"D"``, …).  Auto-inferred
        when ``None``.
    horizon : int, default=3
        Number of future steps to predict.
    lookback : int, default=12
        Window size for window-based models (sklearn / torch).
    split : Split, optional
        Train/val/test split.  Default: ``TimeSplit(train_ratio=0.7, val_ratio=0.15)``.
    transforms : Compose or None, optional
        Default: ``None`` (no transformation).
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        target: str | list[str],
        features: Optional[list[str]] = None,
        time_col: str = "time",
        freq: Optional[str] = None,
        horizon: int = 3,
        lookback: int = 12,
        split: Optional[TimeSplit] = None,
        transforms: Optional[Compose] = None,
    ):
        self.data = data
        self.target = [target] if isinstance(target, str) else target
        self.features = features or self.target  # default: autoregressive
        self.time_col = time_col
        self.horizon = horizon
        self.lookback = lookback
        self.split = split or TimeSplit(train_ratio=0.7, val_ratio=0.15)
        self.transforms = transforms
        self.freq = freq

        # If freq not provided, infer from data
        if self.freq is None:
            sample_times = self._get_time_series()
            inferred = infer_freq(sample_times)
            if inferred:
                self.freq = inferred
                print(f"  [Task] 自动推断频率: {self.freq}  "
                      f"(season_length={self.freq_to_season_length()})")
            else:
                self.freq = "MS"
                print(f"  [Task] 无法推断频率，默认: {self.freq}")

    # ── Helpers ──────────────────────────────────────────────────

    def _get_time_series(self) -> pd.Series:
        """Return the time column as a Series for freq inference."""
        if isinstance(self.data, pd.DataFrame):
            return pd.to_datetime(self.data[self.time_col])
        # CSV — read just the time column
        df = pd.read_csv(self.data, usecols=[self.time_col])
        return pd.to_datetime(df[self.time_col])

    def freq_to_season_length(self) -> int:
        return freq_to_season_length(self.freq)

    def freq_to_time_unit(self) -> str:
        return freq_to_time_unit(self.freq)

    def save_data(self, path: str = "/tmp/_forecast_task_data.csv") -> str:
        """Write data to CSV and return the path (for ForecastPipeline)."""
        if isinstance(self.data, pd.DataFrame):
            self.data.to_csv(path, index=False)
            return path
        return self.data

    # ── Build bundle ─────────────────────────────────────────────

    def build_bundle(self) -> PipelineBundle:
        """Run the data pipeline and return a PipelineBundle with freq set."""
        csv_path = self.save_data()
        pipeline = ForecastPipeline(
            loader=CsvLoader(
                time_col=self.time_col,
                target_cols=self.target,
                feature_cols=self.features,
            ),
            split=self.split,
            transforms=self.transforms,
            window=SlidingWindow(lookback=self.lookback, horizon=self.horizon),
        )
        bundle = pipeline.run(csv_path)
        bundle.freq = self.freq
        return bundle

    # ── Step 1: Compare models ───────────────────────────────────

    def compare(self, models: list[str],
                cache_dir: Optional[str | Path] = None) -> ModelVault:
        """Train multiple models and return a ModelVault with comparison metrics.

        训练结果会自动缓存到 ``cache_dir / f\"{model_name}.pkl\"``。
        当数据或配置发生变化时，缓存自动失效并重新训练。
        若 ``cache_dir=None``，不缓存（每次从头训练）。

        Parameters
        ----------
        models : list of str
            Registered model names (e.g. ``["sf_auto_ets", "xgb"]``).
        cache_dir : str or Path or None, optional
            缓存目录。指定后每个模型的结果会缓存到 ``{cache_dir}/{name}.pkl``，
            下次相同数据+配置时直接加载。

        Returns
        -------
        ModelVault
            Contains trained ``InferencePipeline`` instances, a metrics
            comparison DataFrame, and ``.test_predictions`` dict.
        """
        import hashlib
        import pickle

        bundle = self.build_bundle()
        results: dict[str, TrainResult] = {}
        test_data: dict[str, pd.DataFrame] = {}
        time_col = self.time_col
        test_times = pd.to_datetime(bundle.test_df[self.time_col]) if bundle.test_df is not None else None

        # 计算数据+配置指纹（所有模型共享）
        data_fingerprint = self._compute_data_fingerprint(bundle)

        for name in models:
            cls = get(name)
            season = self.freq_to_season_length()

            # 确定模型参数
            paradigm = cls.paradigm()
            if paradigm in ("torch", "sklearn"):
                model_params = dict(
                    input_dim=bundle.n_features,
                    lookback=self.lookback,
                    horizon=self.horizon,
                    target_dim=len(self.target),
                )
                model = cls(**model_params)
            else:
                model_params = dict(season_length=season)
                model = cls(**model_params)

            # 模型指纹 = 数据指纹 + 模型名 + 模型参数
            model_fingerprint = self._model_fingerprint(
                data_fingerprint, name, model_params
            )

            # ── 缓存检查 ──────────────────────────────────────
            cached = None
            if cache_dir is not None:
                cache_path = Path(cache_dir) / f"{name}.pkl"
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            cached = pickle.load(f)
                        if cached.get("fingerprint") != model_fingerprint:
                            cached = None
                    except Exception:
                        cached = None

            if cached is not None:
                result = cached["result"]
                results[name] = result
                td = cached.get("test_data")
                if td is not None:
                    test_data[name] = td
                m = result.metrics.iloc[0]
                print(f"  {name:30s}  ← 缓存  MAE={m['MAE']:.0f}  "
                      f"R²={m['R2']:.3f}  PearsonR={m['PearsonR']:.3f}")
                continue

            # ── 训练 ──────────────────────────────────────────
            result = EpiAITrainer(model, verbose=False).fit(bundle)
            results[name] = result
            m = result.metrics.iloc[0]
            print(f"  {name:30s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}  "
                  f"PearsonR={m['PearsonR']:.3f}")

            # 测试集预测 DataFrame
            n_pred = result.predictions.shape[0]
            y_pred = result.predictions[:, 0, 0]
            y_true = bundle.get_y_series("test")[:n_pred].ravel()
            times = test_times[:n_pred] if test_times is not None else range(n_pred)
            td = pd.DataFrame({
                time_col: times,
                "y_true": y_true,
                "y_pred": y_pred,
            })
            test_data[name] = td

            # ── 写入缓存 ──────────────────────────────────────
            if cache_dir is not None:
                cache_path = Path(cache_dir) / f"{name}.pkl"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "fingerprint": model_fingerprint,
                        "result": result,
                        "test_data": td,
                    }, f)
                print(f"         → 已缓存至 {cache_path}")

        vault = ModelVault.from_results(results, bundle)
        vault.test_predictions = test_data
        return vault

    def _compute_data_fingerprint(self, bundle) -> str:
        """生成数据+配置的指纹（所有模型共享的部分）。"""
        import hashlib

        h = hashlib.md5()

        # 数据的行列指纹（不 hash 全部值，用 shape + 行 hash）
        for split_name in ("train", "val", "test"):
            df = getattr(bundle, f"{split_name}_df")
            if df is not None:
                h.update(split_name.encode())
                h.update(str(df.shape).encode())
                h.update(str(df.columns.tolist()).encode())
                # 首尾各 3 行 + 随机 3 行
                sample = pd.concat([
                    df.head(3), df.tail(3),
                    df.sample(min(3, len(df)), random_state=42)
                ]).reset_index(drop=True)
                h.update(pd.util.hash_pandas_object(sample).sum())

        # 配置
        h.update(str(bundle.freq).encode())
        h.update(str(self.lookback).encode())
        h.update(str(self.horizon).encode())
        h.update(str(self.features).encode())
        h.update(str(self.target).encode())
        h.update(str(bundle.transforms).encode())
        h.update(str(self.split.train_end).encode())
        h.update(str(self.split.val_end).encode())

        return h.hexdigest()

    def _model_fingerprint(self, data_fp: str, model_name: str,
                           model_params: dict) -> str:
        """数据指纹 + 模型标识 = 模型级指纹。"""
        import hashlib
        h = hashlib.md5(data_fp.encode())
        h.update(model_name.encode())
        h.update(str(sorted(model_params.items())).encode())
        return h.hexdigest()

    # ── Step 2: Deploy ───────────────────────────────────────────

    def deploy(
        self,
        model_names: str | list[str],
        vault: ModelVault,
        risk: Optional[str | list[str] | object | list[object]] = None,
    ) -> DeploymentRuntime:
        """Deploy trained model(s) into a production runtime.

        Parameters
        ----------
        model_names : str or list of str
            Name(s) of the model(s) in the vault to deploy.
        vault : ModelVault
            Vault returned by ``compare()``.
        risk : str, list of str, RiskScorer, list of RiskScorer, optional
            Risk scoring method(s).  Each method gets its own scorer::

                risk="quantile"              → runtime.risk_scorers["quantile"]
                risk=["quantile", "zscore"]  → runtime.risk_scorers["quantile"],
                                                runtime.risk_scorers["zscore"]

        Returns
        -------
        DeploymentRuntime
        """
        bundle = vault.bundle
        runtime = DeploymentRuntime(
            vault=vault,
            time_col=self.time_col,
            time_unit=self.freq_to_time_unit(),
            data_table=self._get_data_for_runtime(),
        )

        # Attach risk scorer(s)
        if risk is not None:
            self._attach_risk(runtime, risk)

        return runtime

    def _get_data_for_runtime(self) -> pd.DataFrame:
        """Get the full dataset as a DataFrame for the runtime data_table."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy()
        return pd.read_csv(self.data)

    def _attach_risk(self, runtime: DeploymentRuntime,
                     risk: str | list[str] | object | list[object]) -> None:
        """Attach one or more RiskScorers to the runtime."""
        from EpiAI.risk import RiskScorer

        # Normalize to list of (name, method_or_scorer)
        risk_list: list[tuple[str, str | RiskScorer]] = []
        items = risk if isinstance(risk, list) else [risk]
        for item in items:
            if isinstance(item, str):
                risk_list.append((item, item))
            elif isinstance(item, RiskScorer):
                # Derive name from the scorer's method attribute
                name = getattr(item, "method", "risk")
                risk_list.append((name, item))
            else:
                raise TypeError(f"Unsupported risk type: {type(item)}")

        for name, spec in risk_list:
            if isinstance(spec, str):
                scorer = RiskScorer(
                    method=spec,
                    time_col=self.time_col,
                    value_col=self.target[0],
                )
            else:
                scorer = spec
            scorer.fit(runtime.data_table)
            runtime.risk_scorers[name] = scorer

    def __repr__(self) -> str:
        return (
            f"ForecastTask(target={self.target}, "
            f"freq={self.freq}, "
            f"horizon={self.horizon})"
        )
