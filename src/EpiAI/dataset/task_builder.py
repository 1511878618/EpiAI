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
    Build x/y/mark tensors from a raw tensor of shape (time, city, feature).

    Output shape convention
    -----------------------
    x:
        (time, city, input_dim)
    y:
        (time, city, target_dim)
    mark:
        (time, city, mark_dim) or None
    """

    def __init__(self, raw_data: DiseaseTensorData) -> None:
        self.raw_data = raw_data

    def build_xy(
        self,
        target_feature_names: list[str],
        input_feature_mode: InputFeatureMode = "exclude_targets",
        input_feature_names: Optional[list[str]] = None,
        mark_feature_names: Optional[list[str]] = None,
        label_len: Optional[int] = None,
        remove_mark_from_input: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
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

        mark_feature_names = list(mark_feature_names or [])

        missing_targets = [name for name in target_feature_names if name not in feature_names]
        if missing_targets:
            raise ValueError(f"Target features not found: {missing_targets}")

        missing_marks = [name for name in mark_feature_names if name not in feature_names]
        if missing_marks:
            raise ValueError(f"Mark features not found: {missing_marks}")

        target_feature_indices = [feature_names.index(name) for name in target_feature_names]
        mark_feature_indices = [feature_names.index(name) for name in mark_feature_names]

        target_set = set(target_feature_indices)
        mark_set = set(mark_feature_indices)

        overlap_target_mark = target_set & mark_set
        if overlap_target_mark:
            overlap_names = [feature_names[i] for i in sorted(overlap_target_mark)]
            raise ValueError(
                f"Target features and mark features must not overlap, got: {overlap_names}"
            )

        # ------------------------------------------------------------------
        # Resolve input features
        # ------------------------------------------------------------------
        if input_feature_mode == "all":
            input_feature_indices = list(range(len(feature_names)))

        elif input_feature_mode == "exclude_targets":
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

        # Optionally remove mark features from input
        if remove_mark_from_input and len(mark_feature_indices) > 0:
            input_feature_indices = [i for i in input_feature_indices if i not in mark_set]

        # Optional: de-duplicate while preserving order
        seen = set()
        input_feature_indices = [i for i in input_feature_indices if not (i in seen or seen.add(i))]

        x = tensor[:, :, input_feature_indices]
        y = tensor[:, :, target_feature_indices]
        mark = tensor[:, :, mark_feature_indices] if len(mark_feature_indices) > 0 else None

        metadata = {
            "province_names": province_names,
            "time_index": time_index,
            "feature_names": feature_names,
            "input_feature_names": [feature_names[i] for i in input_feature_indices],
            "target_feature_names": [feature_names[i] for i in target_feature_indices],
            "mark_feature_names": [feature_names[i] for i in mark_feature_indices],
            "input_feature_indices": input_feature_indices,
            "target_feature_indices": target_feature_indices,
            "mark_feature_indices": mark_feature_indices,
        }

        return x, y, mark, metadata



__all__ = ["FeatureTaskBuilder"]
