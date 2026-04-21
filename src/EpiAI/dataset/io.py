"""
Raw tensor loading from disk.
"""
from __future__ import annotations

import torch

from .containers import DiseaseTensorData


def load_disease_tensor(
    data_path: str = "data/虫媒数据/Align_data_tensor_with_name.pt",
) -> DiseaseTensorData:
    """
    Load raw tensor dataset from disk.
    """
    obj = torch.load(data_path)
    return DiseaseTensorData(
        tensor=obj["tensor"],
        dims=obj["dims"],
        coords=obj["coords"],
    )


__all__ = ["load_disease_tensor"]
