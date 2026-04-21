"""
Utilities to inspect the raw dataset before building forecasting tasks.
"""
from __future__ import annotations

from typing import Optional

import torch

from .containers import DiseaseTensorData
from .io import load_disease_tensor
from .type_defs import InputFeatureMode


class DatasetInspector:
    """
    Inspect raw dataset before building forecasting datasets.
    """

    def __init__(self, raw_data: DiseaseTensorData) -> None:
        self.raw_data = raw_data

    @classmethod
    def from_path(
        cls,
        data_path: str = "data/虫媒数据/Align_data_tensor_with_name.pt",
    ) -> "DatasetInspector":
        return cls(load_disease_tensor(data_path))

    @property
    def tensor(self) -> torch.Tensor:
        return self.raw_data.tensor

    @property
    def dims(self):
        return self.raw_data.dims

    @property
    def coords(self) -> dict:
        return self.raw_data.coords

    def summary(self) -> None:
        print("========== Dataset Summary ==========")
        print(f"Tensor shape: {tuple(self.tensor.shape)}")
        print(f"Dims: {self.dims}")
        for key, value in self.coords.items():
            try:
                length = len(value)
            except TypeError:
                length = "N/A"
            print(f"coords['{key}']: length={length}")
        print("====================================")

    def show_features(self) -> list[str]:
        features = list(self.coords.get("feature", []))
        print("Available features:")
        for i, name in enumerate(features):
            print(f"  [{i}] {name}")
        return features

    def show_provinces(self) -> list[str]:
        provinces = list(self.coords.get("province", []))
        print("Available provinces:")
        for i, name in enumerate(provinces):
            print(f"  [{i}] {name}")
        return provinces

    def show_time_range(self, n_head: int = 5, n_tail: int = 5) -> list:
        time_index = list(self.coords.get("Year/Month", []))
        print("Time index preview:")
        print(f"  Total length: {len(time_index)}")
        if len(time_index) > 0:
            print(f"  Head ({min(n_head, len(time_index))}): {time_index[:n_head]}")
            print(f"  Tail ({min(n_tail, len(time_index))}): {time_index[-n_tail:]}")
        return time_index

    def find_features(self, keyword: str) -> list[str]:
        features = list(self.coords.get("feature", []))
        matched = [f for f in features if keyword.lower() in str(f).lower()]
        print(f"Features containing `{keyword}`:")
        for name in matched:
            print(f"  {name}")
        return matched

    def preview_task_definition(
        self,
        target_feature_names: list[str],
        input_feature_mode: InputFeatureMode = "all",
        input_feature_names: Optional[list[str]] = None,
    ) -> None:
        feature_names = list(self.coords.get("feature", []))

        missing_targets = [f for f in target_feature_names if f not in feature_names]
        if missing_targets:
            raise ValueError(f"Target features not found: {missing_targets}")

        target_indices = [feature_names.index(f) for f in target_feature_names]

        if input_feature_mode == "all":
            input_indices = list(range(len(feature_names)))
        elif input_feature_mode == "exclude_targets":
            target_set = set(target_indices)
            input_indices = [i for i in range(len(feature_names)) if i not in target_set]
        elif input_feature_mode == "explicit":
            if input_feature_names is None:
                raise ValueError("`input_feature_names` must be provided when input_feature_mode='explicit'.")
            missing_inputs = [f for f in input_feature_names if f not in feature_names]
            if missing_inputs:
                raise ValueError(f"Input features not found: {missing_inputs}")
            input_indices = [feature_names.index(f) for f in input_feature_names]
        else:
            raise ValueError(f"Unsupported input_feature_mode: {input_feature_mode}")

        selected_inputs = [feature_names[i] for i in input_indices]

        print("========== Task Definition Preview ==========")
        print(f"Target features ({len(target_feature_names)}):")
        for i, name in zip(target_indices, target_feature_names):
            print(f"  [{i}] {name}")

        print(f"Input feature mode: {input_feature_mode}")
        print(f"Input features ({len(selected_inputs)}):")
        for i in input_indices:
            print(f"  [{i}] {feature_names[i]}")
        print("============================================")

    def quick_preview(self) -> None:
        self.summary()
        self.show_features()
        self.show_provinces()
        self.show_time_range()


__all__ = ["DatasetInspector"]
