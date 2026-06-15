"""
Data transformations for the preprocessing pipeline.

Every transform is a ``Transform`` subclass with optional ``fit``,
required ``transform``, and optional ``inverse``.

``SlidingWindow`` is NOT a ``Transform`` — it converts a 2-D DataFrame
into 3-D numpy arrays and is applied *after* all other transforms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import Transform


# ============================================================================
# Identity (no-op)
# ============================================================================

class Identity(Transform):
    """No transformation — pass-through."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


# ============================================================================
# StandardScaler  —  (x - μ) / σ
# ============================================================================

class StandardScaler(Transform):
    """Z-score standardization.

    Fits mean / std on the *training* split (via ``fit``), then
    applies to any split.

    Parameters
    ----------
    columns : list of str or None
        Columns to standardize.  When *None*, all numeric columns.
    with_mean : bool, optional
        Center before scaling (default ``True``).
    with_std : bool, optional
        Divide by standard deviation (default ``True``).
    eps : float, optional
        Small constant to avoid division by zero (default ``1e-8``).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "StandardScaler":
        cols = self._resolve_columns(df)
        self.mean_ = df[cols].mean()
        self.std_ = df[cols].std()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        df = df.copy()
        if self.with_mean:
            df[cols] = df[cols] - self.mean_
        if self.with_std:
            df[cols] = df[cols] / (self.std_ + self.eps)
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        df = df.copy()
        if self.with_std:
            df[cols] = df[cols] * (self.std_ + self.eps)
        if self.with_mean:
            df[cols] = df[cols] + self.mean_
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return list(df.select_dtypes(include=[np.number]).columns)


# ============================================================================
# RobustScaler  —  (x - median) / IQR
# ============================================================================

class RobustScaler(Transform):
    """Robust standardization using median and IQR.

    Parameters
    ----------
    columns : list of str or None
        Columns to scale.  When *None*, all numeric columns.
    with_centering : bool, optional
        Center by median (default ``True``).
    with_scaling : bool, optional
        Scale by IQR (default ``True``).
    quantile_range : tuple, optional
        Lower / upper quantile for IQR (default ``(25, 75)``).
    eps : float, optional
        Small constant to avoid division by zero (default ``1e-8``).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        eps: float = 1e-8,
    ) -> None:
        self.columns = columns
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.eps = eps
        self.center_: Optional[pd.Series] = None
        self.scale_: Optional[pd.Series] = None

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "RobustScaler":
        cols = self._resolve_columns(df)
        self.center_ = df[cols].median()
        lower = df[cols].quantile(self.quantile_range[0] / 100)
        upper = df[cols].quantile(self.quantile_range[1] / 100)
        self.scale_ = upper - lower
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        df = df.copy()
        if self.with_centering:
            df[cols] = df[cols] - self.center_
        if self.with_scaling:
            df[cols] = df[cols] / (self.scale_ + self.eps)
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        df = df.copy()
        if self.with_scaling:
            df[cols] = df[cols] * (self.scale_ + self.eps)
        if self.with_centering:
            df[cols] = df[cols] + self.center_
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return list(df.select_dtypes(include=[np.number]).columns)


# ============================================================================
# Log1pTransform  —  log(1 + x)
# ============================================================================

class Log1pTransform(Transform):
    """Apply ``log(1 + x)`` transformation.

    Parameters
    ----------
    columns : list of str or None
        Columns to transform.  When *None*, all numeric columns.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        df = df.copy()
        df[cols] = np.log1p(df[cols].clip(lower=0))
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        df = df.copy()
        df[cols] = np.expm1(df[cols])
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return list(df.select_dtypes(include=[np.number]).columns)


# ============================================================================
# BoxCoxTransform  —  Box-Cox power transform
# ============================================================================

class BoxCoxTransform(Transform):
    """Box-Cox power transform.

    Fits the ``λ`` parameter per column using ``scipy.stats.boxcox``.

    Parameters
    ----------
    columns : list of str or None
        Columns to transform.  When *None*, all numeric columns.
    lmbda : float or None, optional
        Fixed lambda.  When *None*, estimate from training data.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        lmbda: Optional[Union[float, Dict[str, float]]] = None,
    ) -> None:
        self.columns = columns
        self.lmbda_input = lmbda
        self.lmbda_: Dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "BoxCoxTransform":
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("BoxCoxTransform requires scipy.")

        if self.lmbda_input is not None and isinstance(self.lmbda_input, dict):
            self.lmbda_ = dict(self.lmbda_input)
            return self

        cols = self._resolve_columns(df)
        for col in cols:
            vals = df[col].clip(lower=1e-8)
            if self.lmbda_input is not None and isinstance(self.lmbda_input, (int, float)):
                self.lmbda_[col] = float(self.lmbda_input)
            else:
                _, lmbda = stats.boxcox(vals)
                self.lmbda_[col] = lmbda
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("BoxCoxTransform requires scipy.")

        cols = self._resolve_columns(df)
        df = df.copy()
        for col in cols:
            if col not in self.lmbda_:
                continue
            vals = df[col].clip(lower=1e-8)
            df[col] = stats.boxcox(vals, lmbda=self.lmbda_[col])
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        df = df.copy()
        for col in cols:
            if col not in self.lmbda_:
                continue
            lmbda = self.lmbda_[col]
            if abs(lmbda) < 1e-8:
                df[col] = np.exp(df[col])
            else:
                df[col] = np.power(df[col] * lmbda + 1, 1.0 / lmbda)
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return list(df.select_dtypes(include=[np.number]).columns)


# ============================================================================
# SelectColumns  — keep / drop columns
# ============================================================================

class SelectColumns(Transform):
    """Keep only specified columns or drop specified columns.

    Parameters
    ----------
    keep : list of str or None, optional
        Columns to keep.
    drop : list of str or None, optional
        Columns to drop.  Ignored when ``keep`` is set.
    """

    def __init__(
        self,
        keep: Optional[List[str]] = None,
        drop: Optional[List[str]] = None,
    ) -> None:
        if keep is not None and drop is not None:
            raise ValueError("Specify either keep or drop, not both.")
        self.keep = keep
        self.drop = drop

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.keep is not None:
            return df[[c for c in self.keep if c in df.columns]]
        if self.drop is not None:
            return df.drop(columns=[c for c in self.drop if c in df.columns])
        return df


# ============================================================================
# DateFeatures  —  extract temporal features from a datetime column
# ============================================================================

class DateFeatures(Transform):
    """Add calendar features extracted from a datetime column.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.
    features : list of str, optional
        Which features to extract.
        Options: ``"year"``, ``"month"``, ``"day"``, ``"dayofweek"``,
        ``"quarter"``, ``"weekofyear"``, ``"is_month_start"``,
        ``"is_month_end"``, ``"season"``.
        Default: ``["year", "month", "dayofweek"]``.
    drop_original : bool, optional
        Remove the original time column (default ``False``).
    """

    _VALID_FEATURES = {
        "year", "month", "day", "dayofweek", "quarter",
        "weekofyear", "is_month_start", "is_month_end", "season",
    }

    SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

    def __init__(
        self,
        time_col: str,
        features: Optional[List[str]] = None,
        drop_original: bool = False,
    ) -> None:
        self.time_col = time_col
        self.features = features or ["year", "month", "dayofweek"]
        unknown = set(self.features) - self._VALID_FEATURES
        if unknown:
            raise ValueError(f"Unknown date features: {unknown}")
        self.drop_original = drop_original

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dt = df[self.time_col]

        for feat in self.features:
            if feat == "season":
                df["season"] = dt.dt.month.map(self.SEASON_MAP)
            elif feat == "weekofyear":
                df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
            elif feat == "is_month_start":
                df["is_month_start"] = dt.dt.is_month_start.astype(int)
            elif feat == "is_month_end":
                df["is_month_end"] = dt.dt.is_month_end.astype(int)
            else:
                df[feat] = getattr(dt.dt, feat)

        if self.drop_original:
            df = df.drop(columns=[self.time_col])

        return df


# ============================================================================
# FeatureLag  —  add lagged versions of specified columns
# ============================================================================

class FeatureLag(Transform):
    """Add lagged values as new columns.

    For multi-entity data, pass ``entity_col`` so lagging is computed
    per entity independently (otherwise lags would leak across entity
    boundaries).

    Parameters
    ----------
    columns : list of str
        Columns to lag.
    lags : list of int
        Lag values (e.g. ``[1, 3, 6]`` for t-1, t-3, t-6).
    entity_col : str or None, optional
        When set, lags are computed per entity group.
    drop_na : bool, optional
        Drop rows with NaN introduced by lagging (default ``True``).
    suffix : str, optional
        Suffix for lag column names (default ``"_lag"``).
    """

    def __init__(
        self,
        columns: List[str],
        lags: List[int],
        entity_col: Optional[str] = None,
        drop_na: bool = True,
        suffix: str = "_lag",
    ) -> None:
        self.columns = columns
        self.lags = sorted(lags)
        self.entity_col = entity_col
        self.drop_na = drop_na
        self.suffix = suffix

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.entity_col is not None and self.entity_col in df.columns:
            parts = []
            for _, group in df.groupby(self.entity_col):
                parts.append(self._add_lags(group))
            df = pd.concat(parts, ignore_index=True)
        else:
            df = self._add_lags(df)
        if self.drop_na:
            df = df.dropna()
        return df

    def _add_lags(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        for col in self.columns:
            if col not in group.columns:
                continue
            for lag in self.lags:
                group[f"{col}{self.suffix}_{lag}"] = group[col].shift(lag)
        return group


# ============================================================================
# SlidingWindow  —  converts 2-D DataFrame → 3-D numpy arrays
# ============================================================================

class SlidingWindow:
    """Convert a flat 2-D DataFrame into sliding-window 3-D arrays.

    This is NOT a ``Transform`` subclass because it changes the data
    shape fundamentally (DataFrame → numpy arrays).  It is applied
    *after* all other transforms and separately to each split.

    Parameters
    ----------
    lookback : int
        Number of past time steps to use as input.
    horizon : int
        Number of future time steps to predict.
    ahead : int, optional
        Gap between input and prediction (default ``0``).
    stride : int, optional
        Step between consecutive windows (default ``1``).
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        ahead: int = 0,
        stride: int = 1,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon
        self.ahead = ahead
        self.stride = stride

    def apply(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        feature_cols: Optional[List[str]] = None,
        time_col: Optional[str] = None,
        entity_col: Optional[str] = None,
    ) -> WindowArrays:
        """Generate sliding windows from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted time-series data.
        target_cols : list of str
            Columns to predict.
        feature_cols : list of str or None
            Columns to use as input.  When *None*, use all numeric
            columns that are not targets or time/entity cols.
        time_col : str or None
            Not used directly; kept for API consistency.
        entity_col : str or None
            When set, windows are generated per entity independently.

        Returns
        -------
        WindowArrays
        """
        if entity_col is not None and entity_col in df.columns:
            return self._apply_by_entity(df, target_cols, feature_cols, entity_col)

        return self._apply_single(df, target_cols, feature_cols)

    def _apply_single(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        feature_cols: Optional[List[str]] = None,
    ) -> "WindowArrays":
        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c not in target_cols]

        X, Y = [], []
        n = len(df)
        for i in range(0, n - self.lookback - self.horizon + 1, self.stride):
            x_end = i + self.lookback
            y_start = x_end + self.ahead
            y_end = y_start + self.horizon
            if y_end > n:
                break
            X.append(df[feature_cols].iloc[i:x_end].values)
            Y.append(df[target_cols].iloc[y_start:y_end].values)

        return WindowArrays(
            x=np.array(X, dtype=np.float32),
            y=np.array(Y, dtype=np.float32),
            feature_names=feature_cols,
            target_names=target_cols,
        )

    def _apply_by_entity(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        feature_cols: Optional[List[str]] = None,
        entity_col: str = "entity",
    ) -> "WindowArrays":
        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c not in target_cols and c != entity_col]

        all_x, all_y = [], []
        for _, group in df.groupby(entity_col):
            group = group.sort_index()
            n = len(group)
            for i in range(0, n - self.lookback - self.horizon + 1, self.stride):
                x_end = i + self.lookback
                y_start = x_end + self.ahead
                y_end = y_start + self.horizon
                if y_end > n:
                    break
                all_x.append(group[feature_cols].iloc[i:x_end].values)
                all_y.append(group[target_cols].iloc[y_start:y_end].values)

        return WindowArrays(
            x=np.array(all_x, dtype=np.float32),
            y=np.array(all_y, dtype=np.float32),
            feature_names=feature_cols,
            target_names=target_cols,
        )


# ============================================================================
# WindowArrays  —  result container for SlidingWindow
# ============================================================================

class WindowArrays:
    """Container for sliding-window output.

    Attributes
    ----------
    x : np.ndarray  — shape ``(n_windows, lookback, n_features)``
    y : np.ndarray  — shape ``(n_windows, horizon, n_targets)``
    feature_names : list of str
    target_names : list of str
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_names: List[str],
    ) -> None:
        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names

    def __repr__(self) -> str:
        return (
            f"WindowArrays(x={self.x.shape}, y={self.y.shape}, "
            f"features={len(self.feature_names)}, targets={len(self.target_names)})"
        )


__all__ = [
    "Identity",
    "StandardScaler",
    "RobustScaler",
    "Log1pTransform",
    "BoxCoxTransform",
    "SelectColumns",
    "DateFeatures",
    "FeatureLag",
    "SlidingWindow",
    "WindowArrays",
]
