"""
ForecastTask — one-shot forecast → compare → full-fit (vault lifecycle).

ModelVault 是贯穿全生命周期的自包含单元：

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

    # 1. compare — 切分评估选型（vault 进入评估态，只存档案）
    vault = task.compare(["sf_auto_ets", "sf_holt_winters", "xgb"])

    # 2. 离线预警评估（可选的阈值方法回测，AUC 等）
    #    vault.evaluate_risk(method="quantile", level=0.90)

    # 3. fit_full — 用全量数据训练最终模型（原地升级到上线态）
    vault.fit_full([vault.best("PearsonR")],
                   risk=["quantile", "zscore"])   # 可同时挂载在线风险评分

    # 4. 在线使用（vault 直接承担运行职责；server 端 load vault 即可）
    report = vault.predict(horizon=6, level=95)
    # → columns: time | model_pred | model_lo | model_hi | model_quantile | ...
    vault.feed(new_df)              # 只追加数据，模型不动
    vault.fit_full(...)             # 需要重训时再全量重训
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

    def build_bundle(self,
                     split: Optional[TimeSplit] = None) -> PipelineBundle:
        """Run the data pipeline and return a PipelineBundle with freq set.

        Parameters
        ----------
        split : TimeSplit or None, optional
            覆盖默认切分。全量训练场景传入
            ``TimeSplit(train_ratio=1.0)``，使 train 覆盖全部数据；
            为 None 时使用任务创建时的 ``self.split``。
        """
        csv_path = self.save_data()
        pipeline = ForecastPipeline(
            loader=CsvLoader(
                time_col=self.time_col,
                target_cols=self.target,
                feature_cols=self.features,
            ),
            split=split if split is not None else self.split,
            transforms=self.transforms,
            window=SlidingWindow(lookback=self.lookback, horizon=self.horizon),
        )
        bundle = pipeline.run(csv_path)
        bundle.freq = self.freq
        return bundle

    # ── Model instantiation ───────────────────────────────────────

    def _make_model(self, name: str, bundle):
        """实例化注册模型，返回 ``(model, model_params)``。

        compare() 与 fit_full() 共用此方法，保证两种流程构造出的
        模型实例与参数指纹完全一致。
        """
        cls = get(name)
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
            model_params = dict(season_length=self.freq_to_season_length())
            model = cls(**model_params)
        return model, model_params

    # ── Step 1: Compare models ───────────────────────────────────

    def compare(self, models: list[str],
                cache_dir: Optional[str | Path] = None,
                keep_eval_weights: bool = False,
                skip_on_error: bool = True) -> ModelVault:
        """Train multiple models and return a ModelVault with comparison metrics.

        训练结果会自动缓存到 ``cache_dir / f\"{model_name}.pkl\"``。
        当数据或配置发生变化时，缓存自动失效并重新训练。
        若 ``cache_dir=None``，不缓存（每次从头训练）。

        Parameters
        ----------
        models : list of str
            Registered model names (e.g. ``[\"sf_auto_ets\", \"xgb\"]``).
        cache_dir : str or Path or None, optional
            缓存目录。指定后每个模型的结果会缓存到 ``{cache_dir}/{name}.pkl``，
            下次相同数据+配置时直接加载。
        keep_eval_weights : bool, default False
            是否在 vault 中保留评估阶段（split 训练）的模型权重。
            默认 False：vault 只保存评估档案（metrics + test_predictions +
            模型配置），权重即用即弃；上线权重由 ``vault.fit_full()`` 产生。
            为 True 时保留（占用空间，可回溯对比）。
        skip_on_error : bool, default True
            单个模型训练失败时跳过该模型、继续其余模型（失败模型不会出现在
            返回 vault 的 metrics/models 中；错误明细挂在 ``vault.compare_failures``）。
            为 False 时保持旧行为：任一模型失败立即抛出。

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
        skipped: list[dict] = []   # 单模型失败记录（skip_on_error=True 时）

        for name in models:
            try:
                model, model_params = self._make_model(name, bundle)

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
            except Exception as e:
                # 单模型训练失败 → 跳过并继续其余模型（不中断整批建档）
                skipped.append({"model": name, "error": f"{type(e).__name__}: {e}"})
                print(f"  ⚠ {name:30s} 训练失败，已跳过 → {type(e).__name__}: {e}")
                continue

        if skipped:
            print(f"  compare: 成功 {len(results)}/{len(models)} 个模型；"
                  f"失败跳过 {len(skipped)} 个: "
                  f"{[s['model'] for s in skipped]}")

        vault = ModelVault.from_results(results, bundle)
        vault.test_predictions = test_data
        # 建档：task 档案 + 全量原始数据；compare = 评估态（运行区默认留空）
        vault.set_task_archive(self, self._get_data_for_runtime())
        vault.phase = "eval"
        vault.metrics_origin = "eval_holdout (compare-phase model)"
        if not keep_eval_weights:
            vault.models = {}
        if skipped:
            vault.compare_failures = skipped   # 瞬态记录，不随 save/load 持久化
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

    # ── Step 1b: Full-data final training ─────────────────────────

    def fit_full(self, models: list[str],
                 cache_dir: Optional[str | Path] = None,
                 eval_vault: Optional[ModelVault] = None,
                 verbose: bool = True,
                 skip_on_error: bool = True) -> ModelVault:
        """用全部数据训练最终上线模型（不划分留出集）。

        与 compare() 的分工：compare() 在 train/val 上训练并用 test 评估，
        用于模型**选择**；fit_full() 把 100% 数据（含最近的观测）一次性
        训练，用于模型**上线**。典型流程::

            vault = task.compare([...])            # 1. 切分评估选型
            vault_full = task.fit_full([vault.best("PearsonR")],
                                       eval_vault=vault)   # 2. 全量训练
            runtime = task.deploy(vault.best("PearsonR"),
                                  vault_full)      # 3. 冻结部署

        Parameters
        ----------
        models : list of str
            要全量训练的模型名（通常取自 compare() 的选型结果）。
        cache_dir : str or Path or None, optional
            缓存目录。每个模型缓存为 ``{cache_dir}/{name}.full.pkl``，
            与 compare() 的 ``{name}.pkl`` 互不干扰、指纹独立。
        eval_vault : ModelVault or None, optional
            compare() 返回的评估 vault。提供后，返回 vault 的 ``.metrics``
            会复制对应模型的留出测试集指标（同超参数、同数据的评估阶段模型
            的指标，供下游 summary()/best()/部署报表使用）；不提供时
            metrics 为 NaN 占位。全量模型自身没有诚实的样本外指标。
        verbose : bool, default True
            是否打印每个模型的训练/缓存状态。
        skip_on_error : bool, default True
            单个模型训练失败时跳过该模型、继续其余模型（失败模型不出现在
            返回 vault 的 models/metrics 中；错误明细挂在 ``vault.fit_failures``）。
            为 False 时保持旧行为：任一模型失败立即抛出。

        Returns
        -------
        ModelVault
            全量训练模型的集合。TS 模型的 ``_last_ds`` 已对齐数据末尾，
            部署后 ``DeploymentRuntime.predict`` 直接从最新观测外推。
        """
        import hashlib
        import pickle

        models = list(dict.fromkeys(models))  # 去重，保持顺序
        bundle = self.build_bundle(split=TimeSplit(train_ratio=1.0))
        data_fingerprint = self._compute_data_fingerprint(bundle)
        results: dict[str, TrainResult] = {}
        skipped: list[dict] = []   # 单模型失败记录（skip_on_error=True 时）

        for name in models:
            try:
                model, model_params = self._make_model(name, bundle)
                # 加 "full|" 前缀，与 compare() 的指纹区分
                model_fingerprint = self._model_fingerprint(
                    "full|" + data_fingerprint, name, model_params
                )

                # ── 缓存检查 ──────────────────────────────────────
                cached = None
                if cache_dir is not None:
                    cache_path = Path(cache_dir) / f"{name}.full.pkl"
                    if cache_path.exists():
                        try:
                            with open(cache_path, "rb") as f:
                                cached = pickle.load(f)
                            if cached.get("fingerprint") != model_fingerprint:
                                cached = None
                        except Exception:
                            cached = None

                if cached is not None:
                    results[name] = cached["result"]
                    if verbose:
                        print(f"  {name:30s}  ← 缓存 (全量)")
                    continue

                # ── 全量训练 ──────────────────────────────────────
                result = EpiAITrainer(model, verbose=False).fit_full(bundle)
                results[name] = result
                if verbose:
                    print(f"  {name:30s}  全量训练完成 (n={len(bundle.train_df)})")

                # ── 写入缓存 ──────────────────────────────────────
                if cache_dir is not None:
                    cache_path = Path(cache_dir) / f"{name}.full.pkl"
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "wb") as f:
                        pickle.dump({
                            "fingerprint": model_fingerprint,
                            "result": result,
                        }, f)
                    if verbose:
                        print(f"         → 已缓存至 {cache_path}")
            except Exception as e:
                # 单模型全量训练失败 → 跳过并继续其余模型
                skipped.append({"model": name, "error": f"{type(e).__name__}: {e}"})
                print(f"  ⚠ {name:30s} 全量训练失败，已跳过 → {type(e).__name__}: {e}")
                continue

        if skipped:
            print(f"  fit_full: 成功 {len(results)}/{len(models)} 个模型；"
                  f"失败跳过 {len(skipped)} 个: "
                  f"{[s['model'] for s in skipped]}")

        vault = ModelVault.from_results(results, bundle)
        vault.test_predictions = {}
        # 建档：task 档案 + 全量原始数据；fit_full = 上线态（运行区 = 全量权重）
        vault.set_task_archive(self, self._get_data_for_runtime())
        vault.phase = "full"
        vault.fitted_on = "full"
        vault.fit_history.append({
            "models": list(vault.models.keys()),
            "time": str(pd.Timestamp.now()),
            "n_rows": len(vault.data_table),
            "last_time": str(vault._history_end_time) if vault._history_end_time else None,
            "phase_prev": "empty",
        })

        # 全量模型无留出指标 → 可继承评估期模型的指标供下游使用
        if eval_vault is not None and not eval_vault.metrics.empty:
            present = [n for n in models if n in eval_vault.metrics.index]
            if present:
                vault.metrics = eval_vault.metrics.loc[present].copy()
                vault.metrics_origin = "eval_holdout (compare-phase model)"
        if skipped:
            vault.fit_failures = skipped   # 瞬态记录，不随 save/load 持久化
        return vault

    # ── Step 2: Deploy ───────────────────────────────────────────

    def deploy(
        self,
        model_names: str | list[str],
        vault: ModelVault,
        risk: Optional[str | list[str] | object | list[object]] = None,
    ) -> ModelVault:
        """[弃用] 冻结上线模型的入口 —— vault 已自含运行能力。

        自 v0.8 起 DeploymentRuntime 职责并入 ModelVault。推荐流程::

            vault = task.compare([...])       # 评估建档
            vault.fit_full([vault.best("PearsonR")])   # 全量训练 → 上线态
            vault.attach_risk([...])          # 可选：在线风险
            report = vault.predict(horizon=6) # 直接预测

        本方法仅为兼容保留：vault 已处于 ``phase="full"`` 时等价于
        ``attach_risk(risk)`` 并返回同一 vault；否则抛出引导错误。
        """
        import warnings
        warnings.warn(
            "ForecastTask.deploy() 已弃用：请使用 vault.fit_full() 升级到上线态后"
            "直接 vault.predict()。本方法将在未来版本移除。",
            DeprecationWarning, stacklevel=2,
        )
        if vault.phase != "full":
            raise ValueError(
                "vault 尚未 fit_full（phase=%r），无法部署。\n"
                "新流程: vault = task.compare([...]) → "
                "vault.fit_full([vault.best(\"PearsonR\")]) → vault.predict(...)" % vault.phase)
        if risk is not None:
            vault.attach_risk(risk)
        return vault

    def _get_data_for_runtime(self) -> pd.DataFrame:
        """Get the full dataset as a DataFrame for the runtime data_table."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy()
        return pd.read_csv(self.data)

    def __repr__(self) -> str:
        return (
            f"ForecastTask(target={self.target}, "
            f"freq={self.freq}, "
            f"horizon={self.horizon})"
        )
