"""
ForecasterRegistry — global model registry.

Usage::

    from EpiAI.models.registry import register, get, list_models

    @register("LSTM", "lstm")
    class LSTMForecaster(nn.Module, TorchMixin):
        ...

    # Usage
    Cls = get("LSTM")               # → LSTMForecaster
    model = Cls(input_dim=8, ...)

    list_models("torch")            # → ["cnn", "lstm", ...]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Type

if TYPE_CHECKING:
    from .base import BaseForecaster

_registry: dict[str, Type["BaseForecaster"]] = {}


def register(*names: str):
    """Decorator that registers a forecaster class under one or more names.

    Example::

        @register("LSTM", "lstm")
        class LSTMForecaster(TorchMixin):
            ...
    """
    def wrapper(cls):
        for name in names:
            _registry[name.lower()] = cls
        return cls
    return wrapper


def get(name: str) -> Type["BaseForecaster"]:
    """Look up a registered forecaster by name (case-insensitive)."""
    cls = _registry.get(name.lower())
    if cls is None:
        available = sorted(_registry.keys())
        raise KeyError(
            f"Unknown model {name!r}. "
            f"Available: {available}"
        )
    return cls


def list_models(paradigm: Optional[str] = None) -> List[str]:
    """Return registered model names, optionally filtered by paradigm.

    Parameters
    ----------
    paradigm : str or None
        One of ``"torch"``, ``"sklearn"``, ``"ts"``, or ``None`` (all).
    """
    if paradigm is None:
        return sorted(_registry.keys())
    return sorted(
        k for k, v in _registry.items()
        if v.paradigm() == paradigm
    )


__all__ = ["register", "get", "list_models"]
