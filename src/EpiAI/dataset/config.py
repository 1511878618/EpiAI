"""
Dataset-level configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .type_defs import InputFeatureMode, SplitMode


@dataclass
class DatasetConfig:
    """
    Dataset-level configuration for city-by-city multi-target forecasting.

    Assumed raw tensor format
    -------------------------
    tensor.shape = (time, city, feature)

    Task definition
    ---------------
    - target_feature_names defines prediction targets
    - input_feature_mode controls how input features are selected

    input_feature_mode choices
    --------------------------
    - "all":
        Use all features as input, including targets.
        Usually not recommended because it may leak target information.
    - "exclude_targets":
        Use all features except target features.
        Recommended default.
    - "explicit":
        Use only `input_feature_names`.
    """
    data_path: str

    target_feature_names: list[str]
    input_feature_mode: InputFeatureMode = "all"
    input_feature_names: Optional[list[str]] = None

    # NEW: mark feature config; 20260421
    mark_feature_names: Optional[list[str]] = None
    remove_mark_from_input: bool = True
    label_len: Optional[int] = None 

    city_dim: int = 1
    split_mode: SplitMode = "cutoff"

    train_val_test_cutoff_line: Optional[tuple[int, int]] = None
    train_city_indices: Optional[list[int]] = None
    val_city_indices: Optional[list[int]] = None
    test_city_indices: Optional[list[int]] = None

    lookback: int = 12
    horizon: int = 3
    ahead: int = 0

    normalize_x: bool = True
    normalize_y: bool = True
    x_norm_dims: tuple[int, ...] = (0, 1)
    y_norm_dims: tuple[int, ...] = (0, 1)

    shuffle_train: bool = True
    shuffle_seed: Optional[int] = 42
    
    # NEW: resolve label_len
    @property
    def resolve_label_len(config: DatasetConfig) -> int:
        if config.label_len is not None:
            if config.label_len <= 0:
                raise ValueError(f"label_len must be > 0, got {config.label_len}")
            if config.label_len > config.lookback:
                raise ValueError(
                    f"label_len ({config.label_len}) must be <= lookback ({config.lookback})"
                )
            return config.label_len

        # default rule
        return config.lookback // 2
__all__ = ["DatasetConfig"]
