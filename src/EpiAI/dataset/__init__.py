"""
disease_forecasting
===================

End-to-end dataset building pipeline for multi-target, city-by-city
time-series forecasting.

Typical usage
-------------
>>> from disease_forecasting import DatasetConfig, MultiTargetCityDatasetBuilder
>>> config = DatasetConfig(
...     data_path="data/虫媒数据/Align_data_tensor_with_name.pt",
...     target_feature_names=["...",],
...     train_val_test_cutoff_line=(20, 27),
... )
>>> bundle = MultiTargetCityDatasetBuilder(config).build()
"""
from __future__ import annotations

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
__all__ = [
    # Lightning Dataset module
    'ForecastDataModule',
    # builder
    "MultiTargetCityDatasetBuilder",
    # config
    "DatasetConfig",
    # containers
    "DiseaseTensorData",
    "CitySplitData",
    "WindowedData",
    "DatasetBundle",
    # inspection
    "DatasetInspector",
    # io
    "load_disease_tensor",
    # normalizer
    "TensorStandardScaler",
    "normalize_split_data",
    # splitter / task builder
    "CitySplitter",
    "FeatureTaskBuilder",
    # typing
    "DimType",
    "InputFeatureMode",
    "SplitMode",
    # utils
    "shuffle_training_data",
    # windowing
    "make_sliding_windows",
    "flatten_city_windows_for_training",
]
