"""
Standardization utilities.

Contains
--------
- TensorStandardScaler : nn.Module-based standardizer with fit/transform API
- normalize_split_data : fit-on-train, transform-all helper for CitySplitData
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .containers import CitySplitData
from .type_defs import DimType


class TensorStandardScaler(nn.Module):
    """
    Standardize a tensor using mean/std computed over specified dimensions.

    Designed for dataset-level normalization:
    - fit on train split only
    - transform train/val/test
    - inverse_transform for evaluation
    """

    def __init__(
        self,
        dims: DimType = (0, 1),
        eps: float = 1e-8,
        unbiased: bool = False,
        clip_std_min: Optional[float] = None,
        check_shape: bool = True,
    ) -> None:
        super().__init__()

        self.dims = self._normalize_dims_config(dims)
        self.eps = float(eps)
        self.unbiased = unbiased
        self.clip_std_min = clip_std_min
        self.check_shape = check_shape

        self.register_buffer("means", None)
        self.register_buffer("stds", None)

    @property
    def fitted(self) -> bool:
        return self.means is not None and self.stds is not None

    def _normalize_dims_config(self, dims: DimType) -> tuple[int, ...]:
        if isinstance(dims, int):
            return (dims,)
        return tuple(dims)

    def _resolve_dims_for_input(self, x: torch.Tensor) -> tuple[int, ...]:
        ndim = x.dim()
        resolved = []

        for d in self.dims:
            if not isinstance(d, int):
                raise TypeError(f"Each dim must be int, got {type(d)}")
            if d < 0:
                d = ndim + d
            if d < 0 or d >= ndim:
                raise ValueError(f"Invalid dim {d} for input with {ndim} dims")
            resolved.append(d)

        if len(set(resolved)) != len(resolved):
            raise ValueError(f"Duplicate dims are not allowed: {resolved}")

        return tuple(sorted(resolved))

    def _expected_stat_shape(
        self,
        x: torch.Tensor,
        dims: tuple[int, ...],
    ) -> tuple[int, ...]:
        shape = list(x.shape)
        for d in dims:
            shape[d] = 1
        return tuple(shape)

    def fit(self, x: torch.Tensor) -> "TensorStandardScaler":
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")

        dims = self._resolve_dims_for_input(x)
        means = x.mean(dim=dims, keepdim=True)
        stds = x.std(dim=dims, keepdim=True, unbiased=self.unbiased)

        if self.clip_std_min is not None:
            stds = stds.clamp_min(self.clip_std_min)

        self.means = means
        self.stds = stds
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")

        if self.check_shape:
            dims = self._resolve_dims_for_input(x)
            expected_stat_shape = self._expected_stat_shape(x, dims)
            if tuple(self.means.shape) != expected_stat_shape:
                raise ValueError(
                    f"Input shape {tuple(x.shape)} is incompatible with fitted statistics. "
                    f"Expected stats shape {expected_stat_shape}, got means shape {tuple(self.means.shape)}."
                )

        return (x - self.means) / (self.stds + self.eps)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        return x * (self.stds + self.eps) + self.means


def normalize_split_data(
    split_data: CitySplitData,
    normalize_x: bool = True,
    normalize_y: bool = True,
    x_norm_dims: tuple[int, ...] = (0, 1),
    y_norm_dims: tuple[int, ...] = (0, 1),
) -> tuple[CitySplitData, Optional[TensorStandardScaler], Optional[TensorStandardScaler]]:
    """
    Fit normalizers on train split only, then transform train/val/test.
    Mark is passed through unchanged.
    """
    x_normalizer = None
    y_normalizer = None

    x_train = split_data.x_train
    y_train = split_data.y_train
    x_val = split_data.x_val
    y_val = split_data.y_val
    x_test = split_data.x_test
    y_test = split_data.y_test
    
    # Currently, mark data is not normalized
    mark_train = split_data.mark_train
    mark_val = split_data.mark_val
    mark_test = split_data.mark_test

    if normalize_x:
        x_normalizer = TensorStandardScaler(dims=x_norm_dims)
        x_normalizer.fit(x_train)
        x_train = x_normalizer.transform(x_train)
        x_val = x_normalizer.transform(x_val)
        x_test = x_normalizer.transform(x_test)

    if normalize_y:
        y_normalizer = TensorStandardScaler(dims=y_norm_dims)
        y_normalizer.fit(y_train)
        y_train = y_normalizer.transform(y_train)
        y_val = y_normalizer.transform(y_val)
        y_test = y_normalizer.transform(y_test)

    normalized_split = CitySplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        mark_train=mark_train,
        mark_val=mark_val,
        mark_test=mark_test,
        train_city_indices=split_data.train_city_indices,
        val_city_indices=split_data.val_city_indices,
        test_city_indices=split_data.test_city_indices,
    )

    return normalized_split, x_normalizer, y_normalizer


__all__ = ["TensorStandardScaler", "normalize_split_data"]
