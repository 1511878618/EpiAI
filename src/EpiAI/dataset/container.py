"""
Unified data containers for the EpiAI data pipeline.

``TimeSeriesData`` is the single currency that every loader produces
and every splitter / transform consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# TimeSeriesData
# ============================================================================

@dataclass
class TimeSeriesData:
    """Core data abstraction, agnostic to the original file format.

    Attributes
    ----------
    df : pd.DataFrame
        Tabular data with at least a time column.
    time_col : str
        Name of the column holding timestamps (parsed as ``datetime``).
    target_cols : List[str]
        Names of the columns to forecast.
    feature_cols : List[str]
        Names of the input feature columns (may overlap with target_cols).
    entity_col : str or None
        When each row belongs to a distinct entity (city / region),
        name of the column holding the entity label.
    entity_values : List[str] or None
        Sorted unique entity labels (e.g. ``["北京", "上海"]``).
    covariates : Dict[str, pd.DataFrame] or None
        Optional extra data frames keyed by name (e.g. weather).
    metadata : dict
        Arbitrary key-value store (original file path, feature
        descriptions, etc.).
    """

    df: pd.DataFrame
    time_col: str
    target_cols: List[str]
    feature_cols: List[str]
    entity_col: Optional[str] = None
    entity_values: Optional[List[str]] = None
    covariates: Optional[Dict[str, pd.DataFrame]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure time column is datetime
        if self.df is not None and self.time_col in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_col]):
                self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
            self.df = self.df.sort_values(self.time_col).reset_index(drop=True)

        # Auto-detect entities
        if self.entity_col is not None and self.entity_col in self.df.columns:
            self.entity_values = sorted(self.df[self.entity_col].unique())

    @property
    def n_entities(self) -> int:
        return len(self.entity_values) if self.entity_values else 1

    @property
    def time_range(self) -> tuple:
        return (self.df[self.time_col].min(), self.df[self.time_col].max())


# ============================================================================
# SplitResult
# ============================================================================

@dataclass
class SplitResult:
    """Holds the result of applying a SplitStrategy.

    Stores **row indices** into the original DataFrame for each split.
    Also stores the actual subset DataFrames for convenience.

    Attributes
    ----------
    train_idx : np.ndarray
        Integer indices of training rows.
    val_idx : np.ndarray
        Integer indices of validation rows.
    test_idx : np.ndarray
        Integer indices of test rows.
    train_data : TimeSeriesData or None
        Subset copy for the training split.
    val_data : TimeSeriesData or None
        Subset copy for the validation split.
    test_data : TimeSeriesData or None
        Subset copy for the test split.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_data: Optional[pd.DataFrame] = None
    val_data: Optional[pd.DataFrame] = None
    test_data: Optional[pd.DataFrame] = None

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


# ============================================================================
# WindowBundle — output of the sliding-window transform
# ============================================================================

@dataclass
class WindowBundle:
    """Container for sliding-window arrays, one per split.

    Each array has shape ``(n_windows, lookback, n_features)`` for
    encoder input and ``(n_windows, horizon, n_targets)`` for decoder
    target.
    """

    train_x: Optional[np.ndarray] = None
    train_y: Optional[np.ndarray] = None
    train_x_mark: Optional[np.ndarray] = None
    train_y_mark: Optional[np.ndarray] = None

    val_x: Optional[np.ndarray] = None
    val_y: Optional[np.ndarray] = None
    val_x_mark: Optional[np.ndarray] = None
    val_y_mark: Optional[np.ndarray] = None

    test_x: Optional[np.ndarray] = None
    test_y: Optional[np.ndarray] = None
    test_x_mark: Optional[np.ndarray] = None
    test_y_mark: Optional[np.ndarray] = None

    x_scaler: Optional[object] = None
    y_scaler: Optional[object] = None


__all__ = [
    "TimeSeriesData",
    "SplitResult",
    "WindowBundle",
]
