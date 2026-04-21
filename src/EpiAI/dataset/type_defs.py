"""
Type aliases shared across the package.
"""
from __future__ import annotations

from typing import Literal, Sequence, Union

DimType = Union[int, Sequence[int], tuple[int, ...]]
SplitMode = Literal["cutoff", "indices"]
InputFeatureMode = Literal["all", "exclude_targets", "explicit"]

__all__ = ["DimType", "SplitMode", "InputFeatureMode"]
