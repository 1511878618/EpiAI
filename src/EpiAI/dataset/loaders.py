"""
Data loaders for common file formats.

Every loader returns a ``TimeSeriesData``, making the rest of the
pipeline format-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .base import DataLoader
from .container import TimeSeriesData


# ============================================================================
# CSV Loader
# ============================================================================

class CsvLoader(DataLoader):
    """Load time-series data from a CSV file.

    Parameters
    ----------
    time_col : str
        Column name holding timestamps.
    target_cols : list of str
        Column(s) to forecast.
    feature_cols : list of str
        Column(s) to use as input features.
    entity_col : str or None, optional
        Column name identifying distinct entities (cities / regions).
        When *None*, all rows belong to a single entity.
    sep : str, optional
        CSV delimiter (default ``","``).
    parse_dates : bool, optional
        Whether to parse the time column as datetime (default ``True``).
    csv_kwargs : dict, optional
        Extra keyword arguments forwarded to ``pd.read_csv``.
    """

    def __init__(
        self,
        time_col: str,
        target_cols: List[str],
        feature_cols: List[str],
        entity_col: Optional[str] = None,
        sep: str = ",",
        parse_dates: bool = True,
        **csv_kwargs,
    ) -> None:
        self.time_col = time_col
        self.target_cols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        self.feature_cols = [feature_cols] if isinstance(feature_cols, str) else list(feature_cols)
        self.entity_col = entity_col
        self.sep = sep
        self.parse_dates = parse_dates
        self.csv_kwargs = csv_kwargs

    def load(self, path: str) -> TimeSeriesData:
        df = pd.read_csv(path, sep=self.sep, **self.csv_kwargs)

        # Validate columns
        missing = [c for c in [self.time_col] + self.target_cols + self.feature_cols
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in CSV: {missing}")

        return TimeSeriesData(
            df=df,
            time_col=self.time_col,
            target_cols=self.target_cols,
            feature_cols=self.feature_cols,
            entity_col=self.entity_col,
            metadata={"loader": "csv", "path": str(path)},
        )


# ============================================================================
# Feather Loader
# ============================================================================

class FeatherLoader(DataLoader):
    """Load time-series data from a Feather / Arrow IPC file.

    Parameters
    ----------
    time_col : str
        Column name holding timestamps.
    target_cols : list of str
        Column(s) to forecast.
    feature_cols : list of str
        Column(s) to use as input features.
    entity_col : str or None, optional
        Column name identifying distinct entities.
    """

    def __init__(
        self,
        time_col: str,
        target_cols: List[str],
        feature_cols: List[str],
        entity_col: Optional[str] = None,
    ) -> None:
        self.time_col = time_col
        self.target_cols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        self.feature_cols = [feature_cols] if isinstance(feature_cols, str) else list(feature_cols)
        self.entity_col = entity_col

    def load(self, path: str) -> TimeSeriesData:
        df = pd.read_feather(path)

        missing = [c for c in [self.time_col] + self.target_cols + self.feature_cols
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in Feather file: {missing}")

        return TimeSeriesData(
            df=df,
            time_col=self.time_col,
            target_cols=self.target_cols,
            feature_cols=self.feature_cols,
            entity_col=self.entity_col,
            metadata={"loader": "feather", "path": str(path)},
        )


# ============================================================================
# Tensor Loader  (legacy .pt 3-D tensor format)
# ============================================================================

class TensorLoader(DataLoader):
    """Load the legacy ``Align_data_tensor_with_name.pt`` format.

    The tensor shape is expected to be ``[time, city, feature]`` with
    a ``coords`` dict describing each axis.

    Parameters
    ----------
    target_feature_names : list of str
        Feature names to predict.
    input_feature_mode : {"all", "exclude_targets", "explicit"}, optional
        How to select input features.
    input_feature_names : list of str or None
        Required when ``input_feature_mode="explicit"``.
    mark_feature_names : list of str or None
        Time-mark features (hour, day, month) kept separate.
    remove_mark_from_input : bool
        Whether to exclude mark features from the input tensor.
    city_dim : int
        Dimension index for cities in the tensor (default ``1``).
    time_label : str
        Label used to find the time coordinate (default ``"Year/Month"``).
    """

    def __init__(
        self,
        target_feature_names: List[str],
        input_feature_mode: str = "all",
        input_feature_names: Optional[List[str]] = None,
        mark_feature_names: Optional[List[str]] = None,
        remove_mark_from_input: bool = True,
        city_dim: int = 1,
        time_label: str = "Year/Month",
        province_label: str = "province",
        feature_label: str = "feature",
    ) -> None:
        self.target_feature_names = target_feature_names if isinstance(target_feature_names, list) else [target_feature_names]
        self.input_feature_mode = input_feature_mode
        self.input_feature_names = input_feature_names or []
        self.mark_feature_names = mark_feature_names or []
        self.remove_mark_from_input = remove_mark_from_input
        self.city_dim = city_dim
        self.time_label = time_label
        self.province_label = province_label
        self.feature_label = feature_label

    def load(self, path: str) -> TimeSeriesData:
        try:
            import torch
        except ImportError:
            raise ImportError("TensorLoader requires PyTorch (torch).")

        obj = torch.load(path, weights_only=False)
        tensor = obj["tensor"]         # (time, city, feature)
        coords: dict = obj["coords"]

        n_time, n_city, n_feat = tensor.shape

        # ── Resolve feature names ────────────────────────────────────
        all_feature_names: List[str] = coords.get(self.feature_label, [])
        if not all_feature_names:
            all_feature_names = [f"feat_{i}" for i in range(n_feat)]

        # Build feature name → index mapping
        feat_name_to_idx = {name: i for i, name in enumerate(all_feature_names)}

        # ── Resolve target indices ────────────────────────────────────
        target_indices = [feat_name_to_idx[n] for n in self.target_feature_names]

        # ── Resolve input feature indices ─────────────────────────────
        if self.input_feature_mode == "all":
            input_indices = list(range(n_feat))
        elif self.input_feature_mode == "exclude_targets":
            input_indices = [i for i in range(n_feat) if i not in set(target_indices)]
        elif self.input_feature_mode == "explicit":
            input_indices = [feat_name_to_idx[n] for n in self.input_feature_names]
        else:
            raise ValueError(f"Unknown input_feature_mode: {self.input_feature_mode}")

        # Optionally remove mark features
        if self.remove_mark_from_input and self.mark_feature_names:
            mark_indices = {feat_name_to_idx[n]
                           for n in self.mark_feature_names if n in feat_name_to_idx}
            input_indices = [i for i in input_indices if i not in mark_indices]

        # ── Resolve time axis ────────────────────────────────────────
        time_labels: List[str] = coords.get(self.time_label, [])
        if not time_labels:
            time_labels = [str(t) for t in range(n_time)]

        # ── Resolve city labels ──────────────────────────────────────
        city_labels: List[str] = coords.get(self.province_label, [])
        if not city_labels:
            city_labels = [f"city_{c}" for c in range(n_city)]

        # ── Expand 3-D tensor → DataFrame ────────────────────────────
        rows = []
        for t in range(n_time):
            for c in range(n_city):
                row = {
                    self.time_label: time_labels[t],
                    self.province_label: city_labels[c],
                }
                for fi in input_indices:
                    row[all_feature_names[fi]] = tensor[t, c, fi].item()
                for fi in target_indices:
                    if fi not in input_indices:
                        row[all_feature_names[fi]] = tensor[t, c, fi].item()
                rows.append(row)

        df = pd.DataFrame(rows)

        # Build feature / target column names
        feature_cols = [all_feature_names[i] for i in input_indices]
        target_cols = [all_feature_names[i] for i in target_indices]

        return TimeSeriesData(
            df=df,
            time_col=self.time_label,
            target_cols=target_cols,
            feature_cols=feature_cols,
            entity_col=self.province_label,
            metadata={
                "loader": "tensor",
                "path": str(path),
                "tensor_shape": list(tensor.shape),
                "all_feature_names": all_feature_names,
                "all_city_names": city_labels,
            },
        )


# ============================================================================
# Auto-detect loader
# ============================================================================

_EXTENSION_MAP: Dict[str, type] = {}


def _register(ext: str, loader_cls: type) -> None:
    _EXTENSION_MAP[ext.lower()] = loader_cls


def load_data(path: str, **kwargs) -> TimeSeriesData:
    """Convenience: auto-detect the loader by file extension.

    For CSV and Feather files you must also pass the column-mapping
    arguments (``time_col``, ``target_cols``, ``feature_cols``, …).

    Parameters
    ----------
    path : str
        File path.
    **kwargs
        Forwarded to the appropriate loader constructor.

    Returns
    -------
    TimeSeriesData
    """
    ext = Path(path).suffix.lower()

    if ext not in _EXTENSION_MAP:
        raise ValueError(
            f"Unsupported extension {ext!r}. "
            f"Supported: {list(_EXTENSION_MAP.keys())}"
        )

    cls = _EXTENSION_MAP[ext]
    loader = cls(**kwargs)
    return loader.load(path)


# Register known extensions
_register(".csv", CsvLoader)
_register(".feather", FeatherLoader)
_register(".pt", TensorLoader)


__all__ = [
    "CsvLoader",
    "FeatherLoader",
    "TensorLoader",
    "load_data",
]
