"""
Train / validation / test splitting strategies.

Every strategy returns a ``SplitResult`` containing the row indices
(and optionally the subset DataFrames) for each split.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import SplitStrategy
from .container import SplitResult, TimeSeriesData


# ============================================================================
# TimeSplit  — split by a time threshold
# ============================================================================

class TimeSplit(SplitStrategy):
    """Split chronologically by time.

    The three simplest patterns:

    * Cutoff mode (default) — one cutoff time separates **train / val**,
      a second separates **val / test**.
    * Ratio mode — split the total time span by proportions.

    Parameters
    ----------
    train_end : str or pd.Timestamp or None, optional
        End of training period (exclusive).  Rows with time < *train_end*
        go to training.  When *None*, infer from ``train_ratio``.
    val_end : str or pd.Timestamp or None, optional
        End of validation period (exclusive).  Rows with
        ``train_end <= time < val_end`` go to validation.
        When *None*, all remaining rows go to test.
    train_ratio : float or None, optional
        Fraction of rows to use for training (alternative to ``train_end``).
        Ignored when ``train_end`` is set.
    val_ratio : float or None, optional
        Fraction of rows to use for validation (alternative to ``val_end``).
        Ignored when ``val_end`` is set.
    """

    def __init__(
        self,
        train_end: Optional[Union[str, pd.Timestamp]] = None,
        val_end: Optional[Union[str, pd.Timestamp]] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
    ) -> None:
        self.train_end = pd.Timestamp(train_end) if train_end else None
        self.val_end = pd.Timestamp(val_end) if val_end else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, data: TimeSeriesData) -> SplitResult:
        df = data.df
        n = len(df)

        # ── Ratio mode: integer-index based ─────────────────────────
        if self.train_ratio is not None:
            n_train = max(1, int(n * self.train_ratio))
            if self.val_ratio is not None:
                n_val = max(0, int(n * self.val_ratio))
                n_test = n - n_train - n_val
            else:
                n_val = n - n_train
                n_test = 0
            train_idx = np.arange(0, n_train)
            val_idx = np.arange(n_train, n_train + n_val)
            test_idx = np.arange(n_train + n_val, n) if n_test > 0 else np.array([], dtype=int)
            return SplitResult(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )

        # ── Date mode: time-threshold based ─────────────────────────
        time = df[data.time_col]

        if self.train_end is None:
            raise ValueError("Either train_end or train_ratio must be provided.")

        train_end = pd.Timestamp(self.train_end)
        train_mask = time < train_end
        train_idx = np.where(train_mask)[0]

        if self.val_end is not None:
            val_end = pd.Timestamp(self.val_end)
            val_mask = (time >= train_end) & (time < val_end)
            test_mask = time >= val_end
        else:
            # No val_end → everything after train_end is validation
            val_mask = time >= train_end
            test_mask = pd.Series([False] * n)

        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        return SplitResult(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )


# ============================================================================
# EntitySplit  — split by entity (city / region)
# ============================================================================

class EntitySplit(SplitStrategy):
    """Split by entity column.  All time points of an entity stay together.

    Parameters
    ----------
    train_entities : list of str
        Entity labels assigned to the training set.
    val_entities : list of str
        Entity labels assigned to the validation set.
    test_entities : list of str
        Entity labels assigned to the test set.
    """

    def __init__(
        self,
        train_entities: List[str],
        val_entities: List[str],
        test_entities: List[str],
    ) -> None:
        self.train_entities = train_entities
        self.val_entities = val_entities
        self.test_entities = test_entities

    def split(self, data: TimeSeriesData) -> SplitResult:
        if data.entity_col is None:
            raise ValueError(
                "EntitySplit requires an entity column. "
                "Set entity_col when creating TimeSeriesData."
            )

        col = data.entity_col
        train_idx = np.where(data.df[col].isin(self.train_entities))[0]
        val_idx = np.where(data.df[col].isin(self.val_entities))[0]
        test_idx = np.where(data.df[col].isin(self.test_entities))[0]

        return SplitResult(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )


# ============================================================================
# EntityTimeSplit  — split per entity *and* per time threshold
# ============================================================================

EntityTimeMap = Dict[str, Tuple[str, str]]


class EntityTimeSplit(SplitStrategy):
    """Each entity gets its own time thresholds.

    Useful when different cities/countries have different outbreak
    timelines and you need per-entity train/val/test windows.

    Parameters
    ----------
    split_map : dict
        ``{entity_label: (train_end, val_end)}`` where each value
        is a 2-tuple of timestamps (or timestamp strings).
        Rows before ``train_end`` → train,
        ``train_end ≤ time < val_end`` → val,
        ``val_end ≤ time`` → test.
    """

    def __init__(self, split_map: EntityTimeMap) -> None:
        self.split_map = {
            k: (pd.Timestamp(v0), pd.Timestamp(v1))
            for k, (v0, v1) in split_map.items()
        }

    def split(self, data: TimeSeriesData) -> SplitResult:
        if data.entity_col is None:
            raise ValueError(
                "EntityTimeSplit requires an entity column."
            )

        df = data.df
        time_col = data.time_col
        entity_col = data.entity_col

        train_idx, val_idx, test_idx = [], [], []

        for entity, (train_end, val_end) in self.split_map.items():
            mask = df[entity_col] == entity
            entity_time = df.loc[mask, time_col]

            train_mask = mask & (entity_time < train_end)
            val_mask = mask & (entity_time >= train_end) & (entity_time < val_end)
            test_mask = mask & (entity_time >= val_end)

            train_idx.extend(np.where(train_mask)[0])
            val_idx.extend(np.where(val_mask)[0])
            test_idx.extend(np.where(test_mask)[0])

        return SplitResult(
            train_idx=np.array(sorted(train_idx)),
            val_idx=np.array(sorted(val_idx)),
            test_idx=np.array(sorted(test_idx)),
        )


# ============================================================================
# CustomIndexSplit  — user provides explicit integer indices
# ============================================================================

class CustomIndexSplit(SplitStrategy):
    """Use pre-computed train / val / test indices (e.g. from a previous
    run, from a file, or from scikit-learn's ``train_test_split``).

    Parameters
    ----------
    train_idx : list of int or np.ndarray
    val_idx : list of int or np.ndarray
    test_idx : list of int or np.ndarray
    """

    def __init__(
        self,
        train_idx: Union[List[int], np.ndarray],
        val_idx: Union[List[int], np.ndarray],
        test_idx: Union[List[int], np.ndarray],
    ) -> None:
        self.train_idx = np.asarray(train_idx)
        self.val_idx = np.asarray(val_idx)
        self.test_idx = np.asarray(test_idx)

    def split(self, data: TimeSeriesData) -> SplitResult:
        n = len(data.df)
        for name, idx in [
            ("train", self.train_idx),
            ("val", self.val_idx),
            ("test", self.test_idx),
        ]:
            if len(idx) == 0:
                continue
            if idx.min() < 0 or idx.max() >= n:
                raise ValueError(
                    f"{name}_idx has values outside [0, {n - 1}]"
                )
        return SplitResult(
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            test_idx=self.test_idx,
        )


# ============================================================================
# NoSplit  — everything is training (for inference / full-fit)
# ============================================================================

class NoSplit(SplitStrategy):
    """Use all data as training (no validation or test)."""

    def split(self, data: TimeSeriesData) -> SplitResult:
        n = len(data.df)
        return SplitResult(
            train_idx=np.arange(n),
            val_idx=np.array([], dtype=int),
            test_idx=np.array([], dtype=int),
        )


# ============================================================================
# CrossValidationSplit — time-series aware cross-validation
# ============================================================================

class CrossValidationSplit(SplitStrategy):
    """Rolling-window cross-validation for time series.

    Each fold uses an expanding or sliding training window followed
    by a fixed-size validation horizon.

    Parameters
    ----------
    n_splits : int, optional
        Number of folds (default 5).
    val_horizon : int, optional
        Number of time steps per validation window (default ``None``:
        inferred as ``len(data) // (n_splits + 1)``).
    expanding : bool, optional
        If ``True`` (default), the training window grows.
        If ``False``, it slides (fixed-size training window).
    train_len : int or None, optional
        Fixed training window length when ``expanding=False``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_horizon: Optional[int] = None,
        expanding: bool = True,
        train_len: Optional[int] = None,
    ) -> None:
        self.n_splits = n_splits
        self.val_horizon = val_horizon
        self.expanding = expanding
        self.train_len = train_len

    def split(self, data: TimeSeriesData) -> SplitResult:
        n = len(data.df)
        val_horizon = self.val_horizon or (n // (self.n_splits + 1))
        train_len = self.train_len or (n - self.n_splits * val_horizon)

        # Only return the *first* fold as the primary split;
        # call .folds() to iterate over all of them.
        val_start = train_len
        val_end = val_start + val_horizon
        train_idx = np.arange(0, train_len)
        val_idx = np.arange(val_start, min(val_end, n))
        test_idx = np.array([], dtype=int)

        return SplitResult(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

    def folds(self, data: TimeSeriesData):
        """Generator yielding one ``SplitResult`` per fold."""
        n = len(data.df)
        val_horizon = self.val_horizon or (n // (self.n_splits + 1))
        train_len = self.train_len or (n - self.n_splits * val_horizon)

        for fold in range(self.n_splits):
            val_start = train_len + fold * val_horizon
            val_end = val_start + val_horizon

            if self.expanding:
                t_start = 0
            else:
                t_start = fold * val_horizon

            train_idx = np.arange(t_start, val_start)
            val_idx = np.arange(val_start, min(val_end, n))
            test_idx = np.array([], dtype=int)

            yield SplitResult(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )


__all__ = [
    "TimeSplit",
    "EntitySplit",
    "EntityTimeSplit",
    "CustomIndexSplit",
    "NoSplit",
    "CrossValidationSplit",
    "EntityTimeMap",
]
