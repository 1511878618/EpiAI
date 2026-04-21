"""
Split x / y tensors along the city dimension.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .containers import CitySplitData
from .type_defs import SplitMode


class CitySplitter:
    """
    Split x/y tensors by city dimension.

    Expected shape convention
    -------------------------
    x : (time, city, input_dim)
    y : (time, city, target_dim)
    """

    def __init__(self, city_dim: int) -> None:
        self.city_dim = city_dim

    def split(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        split_mode: SplitMode,
        train_val_test_cutoff_line: Optional[tuple[int, int]] = None,
        train_city_indices: Optional[Sequence[int]] = None,
        val_city_indices: Optional[Sequence[int]] = None,
        test_city_indices: Optional[Sequence[int]] = None,
    ) -> CitySplitData:
        if split_mode == "cutoff":
            return self._split_by_cutoff(
                x=x,
                y=y,
                train_val_test_cutoff_line=train_val_test_cutoff_line,
            )
        if split_mode == "indices":
            return self._split_by_indices(
                x=x,
                y=y,
                train_city_indices=train_city_indices,
                val_city_indices=val_city_indices,
                test_city_indices=test_city_indices,
            )
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    def _resolve_city_dim(self, x: torch.Tensor) -> int:
        city_dim = self.city_dim
        if city_dim < 0:
            city_dim = x.dim() + city_dim
        if city_dim < 0 or city_dim >= x.dim():
            raise ValueError(f"`city_dim` out of range for x with {x.dim()} dims: got {city_dim}.")
        return city_dim

    def _validate_alignment(self, x: torch.Tensor, y: torch.Tensor, city_dim: int) -> int:
        num_cities = x.shape[city_dim]

        if y.dim() <= city_dim:
            raise ValueError(
                f"`y` does not have city_dim={city_dim}. y.dim()={y.dim()}."
            )

        if y.shape[city_dim] != num_cities:
            raise ValueError(
                f"`x` and `y` are not aligned on city_dim={city_dim}: "
                f"x.shape[{city_dim}]={num_cities}, y.shape[{city_dim}]={y.shape[city_dim]}."
            )

        return num_cities

    def _index_tensor_along_dim(
        self,
        tensor: torch.Tensor,
        dim: int,
        indices: Sequence[int],
    ) -> torch.Tensor:
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
        return torch.index_select(tensor, dim=dim, index=index_tensor)

    def _split_by_cutoff(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_val_test_cutoff_line: Optional[tuple[int, int]],
    ) -> CitySplitData:
        if train_val_test_cutoff_line is None:
            raise ValueError(
                "`train_val_test_cutoff_line` must be provided when split_mode='cutoff'."
            )

        city_dim = self._resolve_city_dim(x)
        num_cities = self._validate_alignment(x, y, city_dim)

        val_start_cutoff, test_start_cutoff = train_val_test_cutoff_line
        if not (0 <= val_start_cutoff <= test_start_cutoff <= num_cities):
            raise ValueError(
                f"Invalid cutoff line {train_val_test_cutoff_line} for num_cities={num_cities}."
            )

        train_indices = list(range(0, val_start_cutoff))
        val_indices = list(range(val_start_cutoff, test_start_cutoff))
        test_indices = list(range(test_start_cutoff, num_cities))

        return self._split_by_indices(
            x=x,
            y=y,
            train_city_indices=train_indices,
            val_city_indices=val_indices,
            test_city_indices=test_indices,
        )

    def _split_by_indices(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_city_indices: Optional[Sequence[int]],
        val_city_indices: Optional[Sequence[int]],
        test_city_indices: Optional[Sequence[int]],
    ) -> CitySplitData:
        if train_city_indices is None or val_city_indices is None or test_city_indices is None:
            raise ValueError(
                "When split_mode='indices', train_city_indices / val_city_indices / "
                "test_city_indices must all be provided."
            )

        city_dim = self._resolve_city_dim(x)
        num_cities = self._validate_alignment(x, y, city_dim)

        train_city_indices = list(train_city_indices)
        val_city_indices = list(val_city_indices)
        test_city_indices = list(test_city_indices)

        all_indices = train_city_indices + val_city_indices + test_city_indices

        for idx in all_indices:
            if not isinstance(idx, int):
                raise ValueError(f"City indices must be integers, got {type(idx)}: {idx}")
            if idx < 0 or idx >= num_cities:
                raise ValueError(
                    f"City index out of range: {idx}. Valid range is [0, {num_cities - 1}]."
                )

        if len(set(all_indices)) != len(all_indices):
            raise ValueError(
                "Duplicated city indices detected across train/val/test splits."
            )

        x_train = self._index_tensor_along_dim(x, city_dim, train_city_indices)
        x_val = self._index_tensor_along_dim(x, city_dim, val_city_indices)
        x_test = self._index_tensor_along_dim(x, city_dim, test_city_indices)

        y_train = self._index_tensor_along_dim(y, city_dim, train_city_indices)
        y_val = self._index_tensor_along_dim(y, city_dim, val_city_indices)
        y_test = self._index_tensor_along_dim(y, city_dim, test_city_indices)

        return CitySplitData(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            train_city_indices=train_city_indices,
            val_city_indices=val_city_indices,
            test_city_indices=test_city_indices,
        )


__all__ = ["CitySplitter"]
