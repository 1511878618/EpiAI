"""Torch forecasting models.

Each model is imported lazily — if dependencies are missing, the
model is skipped so the rest of the module can still load.
"""
from __future__ import annotations

import logging as _logging

_logger = _logging.getLogger(__name__)

_names = [
    ("cnn_lstm", "CNNLSTMForecaster"),
    ("cnn", "CNNForecaster"),
    ("dlinear", "DLinearForecaster"),
    ("lstm", "LSTMForecaster"),
    ("mlp", "MLPForecaster"),
    ("resnet", "ResNetForecaster"),
    ("tcn", "TCNForecaster"),
    ("transformer", "TransformerForecaster"),
    ("Autoformer", "AutoformerForecaster"),
    ("TimesNet", "TimesNetForecaster"),
]

import importlib as _importlib

_loaded = {}
for _mod, _cls in _names:
    try:
        _m = _importlib.import_module(f".{_mod}", __package__)
        _loaded[_cls] = getattr(_m, _cls)
    except ImportError as _e:
        _logger.debug("Skipping %s (%s)", _cls, _e)

# Make all successfully loaded models available at module level
for _k, _v in _loaded.items():
    globals()[_k] = _v

__all__ = list(_loaded.keys())
