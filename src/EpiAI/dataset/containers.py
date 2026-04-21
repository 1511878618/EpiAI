"""
Dataclass containers for every stage of the data pipeline.

Stage map
---------
- DiseaseTensorData   -> raw tensor loaded from disk
- CitySplitData       -> tensors after city-wise split
- WindowedData        -> sliding-window samples
- DatasetBundle       -> final bundle consumed by training / evaluation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    # Avoid a runtime circular import: DatasetBundle only needs this for typing.
    from .normalizer import TensorStandardScaler


@dataclass
class DiseaseTensorData:
    """
    Raw tensor dataset loaded from disk.

    Expected structure
    ------------------
    obj = {
        "tensor": Tensor[time, city, feature],
        "dims": ...,
        "coords": {
            "feature": ...,
            "province": ...,
            "Year/Month": ...,
            ...
        }
    }
    """
    tensor: torch.Tensor
    dims: object
    coords: dict


@dataclass
class CitySplitData:
    """
    Split tensors before sliding-window generation.

    Shapes
    ------
    x_* : (time, city_split, input_dim)
    y_* : (time, city_split, target_dim)
    """
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    train_city_indices: list[int]
    val_city_indices: list[int]
    test_city_indices: list[int]


@dataclass
class WindowedData:
    """
    Sliding-window data.

    Shapes
    ------
    x : (num_samples, lookback, city, input_dim)
    y : (num_samples, horizon,  city, target_dim)
    """
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class DatasetBundle:
    """
    Final dataset bundle for model training and evaluation.
    """
    train_input: torch.Tensor
    train_target: torch.Tensor
    val_input: torch.Tensor
    val_target: torch.Tensor
    test_input: torch.Tensor
    test_target: torch.Tensor

    raw_x: torch.Tensor
    raw_y: torch.Tensor

    split_data: CitySplitData

    x_normalizer: Optional["TensorStandardScaler"]
    y_normalizer: Optional["TensorStandardScaler"]

    city_name_dict: dict[str, list[str]]
    all_city_names: list[str]
    time_index: list

    input_feature_names: list[str]
    target_feature_names: list[str]
    input_feature_indices: list[int]
    target_feature_indices: list[int]


__all__ = [
    "DiseaseTensorData",
    "CitySplitData",
    "WindowedData",
    "DatasetBundle",
]
