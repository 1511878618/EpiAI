"""
Generic backend model adapter.

Wraps any ``.fit(y, X)`` → ``.predict(h, X)`` model into EpiAI's
``TSMixin`` interface in a single class.  Registration becomes one-line::

    register_backend("autotheta", AutoTheta, exogenous=False)
    register_backend("autoarima", AutoARIMA, exogenous=True)

The adapter auto-detects the ``predict`` return format
(dict-with-mean vs raw array).
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Type

import numpy as np

from .base import TSMixin
from .registry import register


# ═════════════════════════════════════════════════════════════════
# Generic adapter
# ═════════════════════════════════════════════════════════════════

class Adapter(TSMixin):
    """Generic TS model adapter.

    Wraps a backend model class whose instances implement::

        model.fit(y: np.ndarray, X: np.ndarray | None = None)
        model.predict(h: int, X: np.ndarray | None = None) -> dict | np.ndarray

    Parameters
    ----------
    backend_cls : type
        The backend model class (e.g. ``statsforecast.models.AutoARIMA``).
    init_kwargs : dict
        Keyword arguments forwarded to ``backend_cls(...)`` at fit time.
    decoder : callable, optional
        How to extract point forecasts from the predict return value.
        Default: ``lambda pred: pred["mean"] if isinstance(pred, dict) else pred``
    clip_negative : bool, default=True
    target_dim_supported : int or ``"n"``, default=1
    exogenous_supported : bool, default=False
    """

    def __init__(
        self,
        backend_cls: type,
        init_kwargs: Optional[dict] = None,
        decoder: Optional[Callable] = None,
        clip_negative: bool = True,
        target_dim_supported: int | Literal["n"] = 1,
        exogenous_supported: bool = False,
    ):
        self._backend_cls = backend_cls
        self._init_kwargs = init_kwargs or {}
        self._decoder = decoder or self._default_decoder
        self.clip_negative = clip_negative
        self.target_dim_supported = target_dim_supported
        self.exogenous_supported = exogenous_supported

        # Runtime state
        self._fitted = None
        self._last_ds = None

    # ── Default decoder ─────────────────────────────────────────

    @staticmethod
    def _default_decoder(pred):
        if isinstance(pred, dict) and "mean" in pred:
            return np.asarray(pred["mean"], dtype=float)
        # Also handle statsforecast's newer format with columns
        if isinstance(pred, dict):
            for key in ("mean", "y_pred", "forecast"):
                if key in pred:
                    return np.asarray(pred[key], dtype=float)
        return np.asarray(pred, dtype=float)

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_1d(arr):
        y = np.asarray(arr, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        return y

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(pred, 0, None) if self.clip_negative else pred

    # ── TSMixin interface ───────────────────────────────────────

    def fit_sequence(self, y_train, X_train=None, **kwargs):
        y = self._to_1d(y_train)
        # Filter init_kwargs to only params the backend actually accepts
        import inspect
        sig = inspect.signature(self._backend_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in self._init_kwargs.items() if k in valid_params}
        model = self._backend_cls(**filtered)
        model.fit(y, X=X_train)
        self._fitted = model
        return self

    def predict_sequence(self, y_test, X_test=None, **kwargs):
        if self._fitted is None:
            raise RuntimeError("not fitted")
        n = len(self._to_1d(y_test))
        raw = self._fitted.predict(h=n, X=X_test)
        return self._clip(self._decoder(raw))

    def forecast(self, n_periods: int, X_future=None):
        if self._fitted is None:
            raise RuntimeError("not fitted")
        raw = self._fitted.predict(h=n_periods, X=X_future)
        pred = self._clip(self._decoder(raw))
        return pred.reshape(-1, 1, 1)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._backend_cls.__name__}, "
            f"exog={self.exogenous_supported})"
        )


# ═════════════════════════════════════════════════════════════════
# One-liner registration helper
# ═════════════════════════════════════════════════════════════════

def register_backend(
    name: str,
    backend_cls: type,
    *,
    aliases: Optional[list[str]] = None,
    defaults: Optional[dict] = None,
    exogenous: bool = False,
    multivariate: bool = False,
    clip_negative: bool = True,
    decoder: Optional[Callable] = None,
):
    """Register any backend model as an EpiAI TS model in one line.

    Parameters
    ----------
    name : str
        Registration name (lowercased).
    backend_cls : type
        Backend class following ``fit(y, X)`` / ``predict(h, X)``.
    aliases : list of str, optional
        Alternative names.
    defaults : dict, optional
        Default init kwargs (e.g. ``dict(season_length=12)``).
        User-supplied kwargs at construction time are merged on top.
    exogenous : bool, default False
        Whether the model accepts ``X``.
    multivariate : bool, default False
        Whether the model can handle multiple targets.
    clip_negative : bool, default True
    decoder : callable, optional
        Custom prediction decoder.

    Examples
    --------
    ::

        register_backend("autotheta", AutoTheta, defaults=dict(season_length=12))
        register_backend("autoarima",  AutoARIMA,  defaults=dict(season_length=12), exogenous=True)
        register_backend("naive",      Naive,      exogenous=False)

        # Usage
        model = get("autotheta")(season_length=6)      # override default
        model = get("naive")()                          # no params needed
    """
    default_kwargs = defaults or {}
    reg_names = [name.lower()] + [a.lower() for a in (aliases or [])]
    target_dim = "n" if multivariate else 1

    class _BackendModel(Adapter):
        target_dim_supported = target_dim
        exogenous_supported = exogenous

        def __init__(self, **kwargs):
            merged = {**default_kwargs, **kwargs}
            super().__init__(
                backend_cls=backend_cls,
                init_kwargs=merged,
                decoder=decoder,
                clip_negative=clip_negative,
                target_dim_supported=target_dim,
                exogenous_supported=exogenous,
            )

    _BackendModel.__name__ = f"Backend_{name}"
    _BackendModel.__qualname__ = _BackendModel.__name__
    _BackendModel.__module__ = __name__

    register(*reg_names)(_BackendModel)

    # Make the class accessible for pickle (pickle needs module-level lookup)
    import sys
    module = sys.modules[__name__]
    setattr(module, _BackendModel.__name__, _BackendModel)

    return _BackendModel
