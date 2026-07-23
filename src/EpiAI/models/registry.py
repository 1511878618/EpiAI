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
    list_models("ts", exogenous=True)  # → TS models that accept X
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Type

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


def list_models(
    paradigm: Optional[str] = None,
    target_dim: Optional[int | Literal["n"]] = None,
    exogenous: Optional[bool] = None,
) -> List[str]:
    """Return registered model names, optionally filtered by paradigm / capabilities.

    Parameters
    ----------
    paradigm : str or None
        One of ``"torch"``, ``"sklearn"``, ``"ts"``, or ``None`` (all).
    target_dim : int, ``"n"``, or None
        If set, only models whose ``target_dim_supported`` matches.
        ``"n"`` matches any multi-variable model.  Non-TS paradigms
        are always excluded from this filter (they handle multi-target
        via their window mechanism).
    exogenous : bool or None
        If set, only models whose ``exogenous_supported`` matches.
    """
    results: list[tuple[str, Type[BaseForecaster]]] = []

    for k, v in _registry.items():
        # ── Paradigm filter ─────────────────────────────────────
        if paradigm is not None and v.paradigm() != paradigm:
            continue

        # ── Capability filters (TS-only; other paradigms pass) ──
        if target_dim is not None:
            if v.paradigm() != "ts":
                continue  # non-TS models handle multi-target natively
            sup = getattr(v, "target_dim_supported", 1)
            if target_dim == "n":
                if sup != "n":
                    continue
            else:
                if sup != "n" and sup != target_dim:
                    continue

        if exogenous is not None:
            if v.paradigm() != "ts":
                continue
            if getattr(v, "exogenous_supported", False) != exogenous:
                continue

        results.append((k, v))

    return sorted(k for k, _ in results)


__all__ = ["register", "get", "list_models"]
