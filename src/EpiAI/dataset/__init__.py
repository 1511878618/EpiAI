"""
EpiAI Dataset Module
====================

**Legacy pipeline** (requires PyTorch):
    DatasetConfig, MultiTargetCityDatasetBuilder, ForecastDataModule, ...

**New pipeline** (pandas/numpy only):
    CsvLoader, FeatherLoader, TensorLoader, TimeSplit, EntitySplit,
    EntityTimeSplit, StandardScaler, RobustScaler, Log1pTransform,
    SlidingWindow, ForecastPipeline, ...

Usage::

    # New pipeline (no torch required)
    from EpiAI.dataset import ForecastPipeline

    bundle = ForecastPipeline.quick(
        path="data.csv",
        time_col="time",
        target_cols="dengue",
        feature_cols=["temp", "humid"],
    )
"""

from __future__ import annotations

# ============================================================================
# New pipeline — zero heavy dependencies
# ============================================================================

from .base import Compose, DataLoader, SplitStrategy, Transform
from .container import SplitResult, TimeSeriesData, WindowBundle
from .loaders import CsvLoader, FeatherLoader, TensorLoader, load_data
from .splits import (
    CrossValidationSplit,
    CustomIndexSplit,
    EntitySplit,
    EntityTimeSplit,
    NoSplit,
    TimeSplit,
)
from .transforms import (
    BoxCoxTransform,
    DateFeatures,
    FeatureLag,
    Identity,
    Log1pTransform,
    RobustScaler,
    SelectColumns,
    SlidingWindow,
    StandardScaler,
    WindowArrays,
)
from .pipeline import ForecastPipeline, PipelineBundle

# ============================================================================
# Legacy pipeline — requires torch
# ============================================================================

# Lazily import torch-dependent components so that the module level
# does not force torch to be installed.
_LEGACY_LOADED = False


def _load_legacy():
    global _LEGACY_LOADED
    if _LEGACY_LOADED:
        return
    from .builder import MultiTargetCityDatasetBuilder
    from .config import DatasetConfig
    from .containers import (
        CitySplitData,
        DatasetBundle,
        DiseaseTensorData,
        WindowedData,
    )
    from .inspector import DatasetInspector
    from .io import load_disease_tensor
    from .normalizer import TensorStandardScaler, normalize_split_data
    from .splitter import CitySplitter
    from .task_builder import FeatureTaskBuilder
    from .type_defs import DimType, InputFeatureMode, SplitMode
    from .utils import shuffle_training_data
    from .windowing import flatten_city_windows_for_training, make_sliding_windows
    from .datamodule import ForecastDataModule

    # Inject into module namespace
    import sys
    mod = sys.modules[__name__]
    for _name, _val in [
        ("MultiTargetCityDatasetBuilder", MultiTargetCityDatasetBuilder),
        ("DatasetConfig", DatasetConfig),
        ("DiseaseTensorData", DiseaseTensorData),
        ("CitySplitData", CitySplitData),
        ("WindowedData", WindowedData),
        ("DatasetBundle", DatasetBundle),
        ("DatasetInspector", DatasetInspector),
        ("load_disease_tensor", load_disease_tensor),
        ("TensorStandardScaler", TensorStandardScaler),
        ("normalize_split_data", normalize_split_data),
        ("CitySplitter", CitySplitter),
        ("FeatureTaskBuilder", FeatureTaskBuilder),
        ("DimType", DimType),
        ("InputFeatureMode", InputFeatureMode),
        ("SplitMode", SplitMode),
        ("shuffle_training_data", shuffle_training_data),
        ("make_sliding_windows", make_sliding_windows),
        ("flatten_city_windows_for_training", flatten_city_windows_for_training),
        ("ForecastDataModule", ForecastDataModule),
    ]:
        setattr(mod, _name, _val)

    _LEGACY_LOADED = True


# Keep backward-compatible aliases so existing code still works.
# Legacy imports will trigger lazy loading on first access, but
# direct access at module level is also supported for explicit use.
# Users can also do: ``from EpiAI.dataset import DatasetConfig``
# which fails with a clear message if torch is missing.

__all__ = [
    # ── New pipeline ──
    "TimeSeriesData",
    "SplitResult",
    "WindowBundle",
    "DataLoader",
    "SplitStrategy",
    "Transform",
    "Compose",
    "CsvLoader",
    "FeatherLoader",
    "TensorLoader",
    "load_data",
    "TimeSplit",
    "EntitySplit",
    "EntityTimeSplit",
    "CustomIndexSplit",
    "NoSplit",
    "CrossValidationSplit",
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
    "ForecastPipeline",
    "PipelineBundle",
]


def __getattr__(name):
    """Lazy-load legacy names for backward compatibility."""
    _legacy = {
        "MultiTargetCityDatasetBuilder",
        "DatasetConfig", "DiseaseTensorData", "CitySplitData",
        "WindowedData", "DatasetBundle", "DatasetInspector",
        "load_disease_tensor", "TensorStandardScaler",
        "normalize_split_data", "CitySplitter", "FeatureTaskBuilder",
        "DimType", "InputFeatureMode", "SplitMode",
        "shuffle_training_data", "make_sliding_windows",
        "flatten_city_windows_for_training", "ForecastDataModule",
    }
    if name in _legacy:
        _load_legacy()
        import sys
        mod = sys.modules[__name__]
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
