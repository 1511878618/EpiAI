"""
ForecastPipeline — high-level orchestrator.

Combines a DataLoader, SplitStrategy, optional transforms, and
SlidingWindow into a single ``run()`` call.

Usage::

    pipeline = ForecastPipeline(
        loader=CsvLoader(time_col="time", target_cols="dengue",
                         feature_cols=["temp", "humid"]),
        split=TimeSplit(train_ratio=0.6, val_ratio=0.2),
        transforms=Compose([
            Log1pTransform(columns=["dengue"]),
            StandardScaler(),
        ]),
        window=SlidingWindow(lookback=12, horizon=3),
    )
    bundle = pipeline.run("data.csv")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import Compose, SplitStrategy, Transform
from .container import SplitResult, TimeSeriesData
from .loaders import DataLoader
from .transforms import SlidingWindow, WindowArrays


# ============================================================================
# PipelineBundle — the output of ForecastPipeline.run()
# ============================================================================

@dataclass
class PipelineBundle:
    """Everything produced by a full pipeline run.

    .. rubric:: Windowed data (transform space, for model training/eval)

    ==========  =================================================
    ``train_x``  ``(n_windows, lookback, n_features)``
    ``train_y``  ``(n_windows, horizon, n_targets)``
    ``val_x``
    ``val_y``
    ``test_x``
    ``test_y``
    ==========  =================================================

    .. rubric:: Flat DataFrames (transform space, for TS models)

    =============  ==============================================
    ``train_df``   Transformed training DataFrame (not windowed).
    ``val_df``
    ``test_df``
    =============  ==============================================

    .. rubric:: Metadata

    ================  ============================================
    ``feature_names``  List of input feature column names.
    ``target_names``   List of target column names.
    ``transforms``     Fitted ``Compose`` (or ``None``).
    ================  ============================================

    .. rubric:: Convenience properties

    ``n_train``  ``n_val``  ``n_test``  ``lookback``
    ``horizon``  ``n_features``  ``n_targets``

    .. rubric:: Raw series access (flat, transform space)

    ``get_y_series(split)``   y in transform space  — for TS models.
    ``get_X_series(split)``   X in transform space.

    .. rubric:: Raw series access (flat, original space)

    ``get_y_series_inverse(split)``   y back in observation units.
    ``get_X_series_inverse(split)``   X back in observation units.
    """

    data: TimeSeriesData
    split: SplitResult

    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: Optional[np.ndarray] = None
    test_y: Optional[np.ndarray] = None

    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)

    transforms: Optional[Compose] = None

    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None

    # ── Utility methods for TimeSeries (ARIMA/ETS) models ──────────

    def get_y_series(self, split: str = "train") -> np.ndarray:
        """Return the transformed raw time series ``(T, n_targets)``.

        Unlike ``train_y`` (which is windowed into 3-D), this returns
        the **flat** sequence.  Used by TS-paradigm models.
        """
        df = getattr(self, f"{split}_df")
        if df is None:
            raise ValueError(f"No {split}_df available; pipeline may not have saved it.")
        return df[self.target_names].values.astype(np.float32)

    def get_y_series_inverse(self, split: str = "train") -> np.ndarray:
        """Return the inverse-transformed raw time series ``(T, n_targets)``.

        Unlike ``get_y_series()`` (which returns values in transform space),
        this returns values in original (observation) space by calling
        ``transforms.inverse()`` on each target column.

        Safe for repeated calls — never modifies internal state.
        """
        y = self.get_y_series(split).copy()
        if self.transforms is None:
            return y
        for i, tn in enumerate(self.target_names):
            inv_series = pd.Series(y[:, i], name=tn)
            inv_df = self.transforms.inverse(inv_series.to_frame())
            y[:, i] = inv_df[tn].values
        return y

    def get_X_series(self, split: str = "train") -> np.ndarray:
        """Return the transformed raw feature series ``(T, n_features)``."""
        df = getattr(self, f"{split}_df")
        if df is None:
            raise ValueError(f"No {split}_df available; pipeline may not have saved it.")
        return df[self.feature_names].values.astype(np.float32)

    def get_X_series_inverse(self, split: str = "train") -> np.ndarray:
        """Return the inverse-transformed raw feature series ``(T, n_features)``."""
        x = self.get_X_series(split).copy()
        if self.transforms is None:
            return x
        for i, fn in enumerate(self.feature_names):
            inv_series = pd.Series(x[:, i], name=fn)
            inv_df = self.transforms.inverse(inv_series.to_frame())
            x[:, i] = inv_df[fn].values
        return x

    @property
    def n_train(self) -> int:
        return len(self.train_x)

    @property
    def n_val(self) -> int:
        return len(self.val_x)

    @property
    def n_test(self) -> int:
        return len(self.test_x) if self.test_x is not None else 0

    @property
    def lookback(self) -> int:
        return self.train_x.shape[1] if len(self.train_x) > 0 else 0

    @property
    def horizon(self) -> int:
        return self.train_y.shape[1] if len(self.train_y) > 0 else 0

    @property
    def n_features(self) -> int:
        return self.train_x.shape[2] if len(self.train_x) > 0 else 0

    @property
    def n_targets(self) -> int:
        return self.train_y.shape[2] if len(self.train_y) > 0 else 0

    def __repr__(self) -> str:
        return (
            f"PipelineBundle(train=({self.n_train}, {self.lookback}, {self.n_features}), "
            f"val=({self.n_val}, {self.lookback}, {self.n_features}), "
            f"test=({self.n_test}, …))"
        )


# ============================================================================
# ForecastPipeline
# ============================================================================

class ForecastPipeline:
    """End-to-end data pipeline: load → split → transform → window.

    Parameters
    ----------
    loader : DataLoader
        How to read the raw data.
    split : SplitStrategy
        How to split into train/val/test.
    transforms : Transform or list of Transform or None, optional
        Transforms to apply *before* windowing.  Pass a ``Compose`` or
        a list (will be wrapped in ``Compose`` automatically).
    window : SlidingWindow or None, optional
        Sliding window configuration.  When *None*, windowing is skipped
        and the pipeline returns the transformed DataFrames instead.
    """

    def __init__(
        self,
        loader: DataLoader,
        split: SplitStrategy,
        transforms: Optional[Transform] = None,
        window: Optional[SlidingWindow] = None,
    ) -> None:
        self.loader = loader
        self.split_strategy = split
        self.window = window

        # Normalize transforms
        if isinstance(transforms, list):
            self.transforms = Compose(transforms)
        else:
            self.transforms = transforms

    # ── Primary entry point ──────────────────────────────────────────

    def run(self, path: str) -> PipelineBundle:
        """Load, split, transform, and window the data."""
        data = self.loader.load(path)
        split = self.split_strategy.split(data)

        # ---- 1. Extract split DataFrames ----
        train_df = data.df.iloc[split.train_idx].copy()
        val_df = data.df.iloc[split.val_idx].copy() if len(split.val_idx) > 0 else None
        test_df = data.df.iloc[split.test_idx].copy() if len(split.test_idx) > 0 else None

        # ---- 2. Fit transforms on train, apply to all ----
        if self.transforms is not None:
            original_feature_cols = set(data.feature_cols)
            original_all_cols = set(data.df.columns)

            self.transforms.fit(
                train_df,
                target_cols=data.target_cols,
                feature_cols=data.feature_cols,
            )
            train_df = self.transforms.transform(train_df)
            if val_df is not None:
                val_df = self.transforms.transform(val_df)
            if test_df is not None:
                test_df = self.transforms.transform(test_df)

            # Resolve actual feature columns after transforms:
            #   original feature cols that still exist
            #   + any NEW numeric columns added by transforms
            #   - time / entity / target columns
            #   Exception: if a column is both feature AND target (autoregressive),
            #   it stays as a feature — past target values are valid input.
            _exclude = set(data.target_cols)
            if data.time_col in train_df.columns:
                _exclude.add(data.time_col)
            if data.entity_col and data.entity_col in train_df.columns:
                _exclude.add(data.entity_col)
            # Don't exclude columns that were explicitly requested as features
            # even if they're also targets (autoregressive overlap).
            _exclude -= set(data.feature_cols or [])

            current_numeric = set(
                train_df.select_dtypes(include=[np.number]).columns
            )
            resolved_feature_cols = [
                c for c in train_df.columns
                if c in current_numeric
                and c not in _exclude
                and (
                    c in original_feature_cols       # kept from original
                    or c not in original_all_cols     # new column from a transform
                )
            ]
        else:
            resolved_feature_cols = data.feature_cols

        # ---- 3. Windowing (optional) ----
        if self.window is None:
            # Return DataFrames directly (no windowing)
            return _bundle_from_dfs(
                data=data,
                split=split,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=resolved_feature_cols,
                transforms=self.transforms,
            )

        train_w = self.window.apply(
            train_df, target_cols=data.target_cols,
            feature_cols=resolved_feature_cols,
            entity_col=data.entity_col,
        )

        # Val/test windows: prepend context from previous split(s).
        # Context must be at least (lookback + horizon - 1) so that
        # the first horizon step (index 0) covers ALL test/val rows.
        context_len = max(self.window.lookback + self.window.horizon - 1, 1)

        if val_df is not None:
            _val_ctx = train_df.iloc[-context_len:] if len(train_df) >= context_len else train_df
            val_w = self.window.apply(
                pd.concat([_val_ctx, val_df]),
                target_cols=data.target_cols,
                feature_cols=resolved_feature_cols,
                entity_col=data.entity_col,
            )
            # Trim windows whose first horizon step lands in the context
            # (not in the actual val_df), so test_y[:,0,:] aligns to val_df[0].
            _trim = context_len - self.window.lookback
            if _trim > 0 and val_w.x.shape[0] > _trim:
                val_w = WindowArrays(
                    x=val_w.x[_trim:], y=val_w.y[_trim:],
                    feature_names=val_w.feature_names,
                    target_names=val_w.target_names,
                )
        else:
            val_w = WindowArrays(
                x=np.empty((0, train_w.x.shape[1], train_w.x.shape[2])),
                y=np.empty((0, train_w.y.shape[1], train_w.y.shape[2])),
                feature_names=train_w.feature_names,
                target_names=train_w.target_names,
            )

        if test_df is not None:
            _prev_for_test = train_df if val_df is None else pd.concat([train_df, val_df])
            _test_ctx = _prev_for_test.iloc[-context_len:] if len(_prev_for_test) >= context_len else _prev_for_test
            test_w = self.window.apply(
                pd.concat([_test_ctx, test_df]),
                target_cols=data.target_cols,
                feature_cols=resolved_feature_cols,
                entity_col=data.entity_col,
            )
            _trim = context_len - self.window.lookback
            if _trim > 0 and test_w.x.shape[0] > _trim:
                test_w = WindowArrays(
                    x=test_w.x[_trim:], y=test_w.y[_trim:],
                    feature_names=test_w.feature_names,
                    target_names=test_w.target_names,
                )
        else:
            test_w = None

        return PipelineBundle(
            data=data,
            split=split,
            train_x=train_w.x,
            train_y=train_w.y,
            val_x=val_w.x,
            val_y=val_w.y,
            test_x=test_w.x if test_w is not None else None,
            test_y=test_w.y if test_w is not None else None,
            feature_names=train_w.feature_names,
            target_names=train_w.target_names,
            transforms=self.transforms,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    # ── Convenience: quick one-shot without boilerplate ──────────────

    @classmethod
    def quick(
        cls,
        path: str,
        time_col: str,
        target_cols,
        feature_cols,
        entity_col: Optional[str] = None,
        lookback: int = 12,
        horizon: int = 3,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalize: bool = True,
    ) -> PipelineBundle:
        """Quick-start pipeline with sensible defaults.

        Parameters
        ----------
        path : str
            Path to CSV file.
        time_col : str
            Time column name.
        target_cols : list of str
            Target column(s).
        feature_cols : list of str
            Feature column(s).
        entity_col : str or None, optional
            Entity column for multi-city data.
        lookback : int
            Lookback window.
        horizon : int
            Prediction horizon.
        train_ratio : float
            Fraction for training.
        val_ratio : float
            Fraction for validation.
        normalize : bool
            Whether to apply StandardScaler.

        Returns
        -------
        PipelineBundle
        """
        from .loaders import CsvLoader
        from .splits import TimeSplit
        from .transforms import StandardScaler

        # Normalize str → list
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        if isinstance(feature_cols, str):
            feature_cols = [feature_cols]

        transforms = [StandardScaler()] if normalize else None
        if transforms:
            transforms = Compose(transforms)

        pipeline = cls(
            loader=CsvLoader(
                time_col=time_col,
                target_cols=target_cols,
                feature_cols=feature_cols,
                entity_col=entity_col,
            ),
            split=TimeSplit(train_ratio=train_ratio, val_ratio=val_ratio),
            transforms=transforms,
            window=SlidingWindow(lookback=lookback, horizon=horizon),
        )
        return pipeline.run(path)


# ============================================================================
# Internal helpers
# ============================================================================

def _bundle_from_dfs(
    data: TimeSeriesData,
    split: SplitResult,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    transforms: Optional[Compose],
) -> PipelineBundle:
    """Create a PipelineBundle with DataFrame output (no windowing)."""
    # Convert DataFrames to arrays (2D → 3D with dummy lookback=1)
    def _to_3d(df, cols):
        if df is None:
            return None
        arr = df[cols].values.astype(np.float32)
        return arr.reshape(arr.shape[0], 1, arr.shape[1])  # (N, 1, D)

    feat = feature_cols
    tgt = data.target_cols

    return PipelineBundle(
        data=data,
        split=split,
        train_x=_to_3d(train_df, feat),
        train_y=_to_3d(train_df, tgt),
        val_x=_to_3d(val_df, feat),
        val_y=_to_3d(val_df, tgt),
        test_x=_to_3d(test_df, feat),
        test_y=_to_3d(test_df, tgt),
        feature_names=feat,
        target_names=tgt,
        transforms=transforms,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


__all__ = [
    "ForecastPipeline",
    "PipelineBundle",
]
