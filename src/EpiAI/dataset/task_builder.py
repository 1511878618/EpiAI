"""
Build x/y tensors from the feature dimension of a raw (time, city, feature) tensor.
"""
from __future__ import annotations

from typing import Optional

import torch

from .containers import DiseaseTensorData
from .type_defs import InputFeatureMode


class FeatureTaskBuilder:
    """
    Build x/y tensors from a raw tensor of shape (time, city, feature).

    Output shape convention
    -----------------------
    x : (time, city, input_dim)
    y : (time, city, target_dim)

    Notes
    -----
    - y is always kept 3D, even when target_dim == 1.
    - input features are selected by mode:
        * all
        * exclude_targets
        * explicit
    """

    def __init__(self, raw_data: DiseaseTensorData) -> None:
        self.raw_data = raw_data

    def build_xy(
        self,
        target_feature_names: list[str],
        input_feature_mode: InputFeatureMode = "exclude_targets",
        input_feature_names: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        tensor = torch.as_tensor(self.raw_data.tensor, dtype=torch.float32)
        coords = self.raw_data.coords

        feature_names = list(coords["feature"])
        province_names = list(coords["province"])
        time_index = list(coords["Year/Month"])

        if tensor.dim() != 3:
            raise ValueError(
                f"Expected raw tensor shape (time, city, feature), got {tuple(tensor.shape)}."
            )

        if len(target_feature_names) == 0:
            raise ValueError("`target_feature_names` must contain at least one feature.")

        missing_targets = [name for name in target_feature_names if name not in feature_names]
        if missing_targets:
            raise ValueError(f"Target features not found: {missing_targets}")

        target_feature_indices = [feature_names.index(name) for name in target_feature_names]

        if input_feature_mode == "all":
            input_feature_indices = list(range(len(feature_names)))

        elif input_feature_mode == "exclude_targets":
            target_set = set(target_feature_indices)
            input_feature_indices = [i for i in range(len(feature_names)) if i not in target_set]

        elif input_feature_mode == "explicit":
            if input_feature_names is None or len(input_feature_names) == 0:
                raise ValueError(
                    "`input_feature_names` must be provided when input_feature_mode='explicit'."
                )
            missing_inputs = [name for name in input_feature_names if name not in feature_names]
            if missing_inputs:
                raise ValueError(f"Input features not found: {missing_inputs}")
            input_feature_indices = [feature_names.index(name) for name in input_feature_names]

        else:
            raise ValueError(f"Unsupported input_feature_mode: {input_feature_mode}")

        x = tensor[:, :, input_feature_indices]
        y = tensor[:, :, target_feature_indices]

        metadata = {
            "province_names": province_names,
            "time_index": time_index,
            "feature_names": feature_names,
            "input_feature_names": [feature_names[i] for i in input_feature_indices],
            "target_feature_names": [feature_names[i] for i in target_feature_indices],
            "input_feature_indices": input_feature_indices,
            "target_feature_indices": target_feature_indices,
        }

        return x, y, metadata


__all__ = ["FeatureTaskBuilder"]
