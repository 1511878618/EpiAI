"""
Abstract base classes for the EpiAI data pipeline.

Defines the contract for all pluggable components:
    DataLoader  — loads data into a unified TimeSeriesData container
    SplitStrategy — splits data into train/val/test
    Transform  — transforms DataFrames (normalize, log, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd

from .container import TimeSeriesData, SplitResult


# ============================================================================
# DataLoader
# ============================================================================

class DataLoader(ABC):
    """Load data from a file into a TimeSeriesData container."""

    @abstractmethod
    def load(self, path: str) -> TimeSeriesData:
        ...


# ============================================================================
# SplitStrategy
# ============================================================================

class SplitStrategy(ABC):
    """Produce train / val / test index masks or direct subsets."""

    @abstractmethod
    def split(self, data: TimeSeriesData) -> SplitResult:
        ...


# ============================================================================
# Transform
# ============================================================================

class Transform(ABC):
    """A single data transformation that operates on a DataFrame.

    Fits on the **training** split, then applies to any split.
    Subclasses must implement ``_fit_to_df`` and ``transform``.
    Optionally override ``inverse`` for reversible transforms.
    """

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "Transform":
        """Fit parameters on the *training* DataFrame.

        The default implementation is a no-op.  Override when the
        transform needs to learn statistics (mean, std, …).
        """
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation; return a new DataFrame."""
        ...

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation (only required for reversible
        transforms such as scalers used on the target variable)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support inverse."
        )


# ============================================================================
# Helpers
# ============================================================================

class Compose(Transform):
    """Chain multiple Transforms and apply them in order."""

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "Compose":
        for t in self.transforms:
            t.fit(df, target_cols, feature_cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in self.transforms:
            df = t.transform(df)
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in reversed(self.transforms):
            df = t.inverse(df)
        return df


__all__ = [
    "DataLoader",
    "SplitStrategy",
    "Transform",
    "Compose",
]
