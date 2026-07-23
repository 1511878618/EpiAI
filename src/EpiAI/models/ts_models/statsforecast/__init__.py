"""StatsForecast-backed model wrappers for EpiAI."""
from __future__ import annotations

from .autoarima import StatsAutoARIMA
from .autoets import StatsAutoETS

__all__ = [
    "StatsAutoARIMA",
    "StatsAutoETS",
]
