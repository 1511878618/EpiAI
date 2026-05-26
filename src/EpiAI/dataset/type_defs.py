"""
Type aliases shared across the package.
"""
from __future__ import annotations

from typing import Literal, Sequence, Union

DimType = Union[int, Sequence[int], tuple[int, ...]]
InputFeatureMode = Literal["all", "exclude_targets", "explicit"]
SplitMode = Literal["cutoff", "indices"]
SplitBy = Literal["time", "city"]


__all__ = ["DimType", "SplitMode", "InputFeatureMode", "SplitBy"]
