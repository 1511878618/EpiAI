"""
EpiAITrainer — unified training entry point.

Automatically routes to the correct training strategy based on
the model's ``paradigm()``:

- ``"torch"``    → epoch loop, AdamW, EarlyStopping, LR scheduler
- ``"sklearn"``  → one-shot ``model.fit(train_x, train_y, ...)``
- ``"ts"``       → ``model.fit_sequence(y_train)`` + ``predict_sequence(y_test)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from EpiAI.models.base import BaseForecaster


# =====================================================================
# TrainResult
# =====================================================================

@dataclass
class TrainResult:
    """Unified training output across all model families.

    Attributes
    ----------
    model : BaseForecaster
        The trained model instance.
    predictions : np.ndarray
        Test-set predictions, shape ``(N, horizon, target_dim)``,
        **inverse-transformed** back to original scale.
    metrics : pd.DataFrame
        Per-target, per-split evaluation metrics
        (MAE, RMSE, MAPE, R², PearsonR).
    history : dict or None
        Training history (loss curves for torch models).
    """

    model: BaseForecaster
    predictions: np.ndarray
    metrics: pd.DataFrame
    history: Optional[dict] = None

    def __repr__(self) -> str:
        return (
            f"TrainResult(model={type(self.model).__name__}, "
            f"preds={self.predictions.shape}, "
            f"metrics={self.metrics.shape})"
        )


# =====================================================================
# EpiAITrainer
# =====================================================================

class EpiAITrainer:
    """Unified trainer that auto-routes by model paradigm.

    Parameters
    ----------
    model : BaseForecaster
        An instantiated model (any paradigm).
    loss : torch.nn.Module or None, optional
        Loss function.  Only used for ``"torch"`` models.
        Sklearn and TS models ignore this parameter.
    optimizer_config : dict or None, optional
        Optimizer kwargs (``lr``, ``weight_decay``, …).
        Only used for ``"torch"`` models.
    early_stopping_config : dict or None, optional
        Early-stopping kwargs (``patience``, ``min_delta``, …).
        Only used for ``"torch"`` models.
    device : str, optional
        Torch device (``"auto"``, ``"cuda"``, ``"cpu"``).
        Only used for ``"torch"`` models.
    verbose : bool, optional
        Print progress during training.
    """

    def __init__(
        self,
        model: BaseForecaster,
        loss=None,
        optimizer_config: Optional[dict] = None,
        early_stopping_config: Optional[dict] = None,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer_config = optimizer_config or {}
        self.early_stopping_config = early_stopping_config or {}
        self.device = device
        self.verbose = verbose

    # ── Public entry point ──────────────────────────────────────────

    def fit(self, bundle) -> TrainResult:
        """Train the model using data from a ``PipelineBundle``.

        Parameters
        ----------
        bundle : PipelineBundle
            Output of ``ForecastPipeline.run()``.

        Returns
        -------
        TrainResult
        """
        paradigm = self.model.paradigm()

        if paradigm == "torch":
            return self._fit_torch(bundle)
        elif paradigm == "sklearn":
            return self._fit_sklearn(bundle)
        elif paradigm == "ts":
            return self._fit_ts(bundle)
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")

    # ── Torch path ──────────────────────────────────────────────────

    def _fit_torch(self, bundle):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required for torch-paradigm models.")

        device = self._resolve_device()
        self._history = {"train_loss": [], "val_loss": []}

        model = self.model.to(device)
        loss_fn = self.loss or nn.MSELoss()

        # Split optimizer_config into AdamW params vs training config
        _opt_keys = {"lr", "weight_decay", "betas", "eps", "amsgrad"}
        _opt_kw = {k: v for k, v in self.optimizer_config.items() if k in _opt_keys}
        optimizer = torch.optim.AdamW(
            model.parameters(), **_opt_kw if _opt_kw else {"lr": 1e-3}
        )
        early_stopper = self._make_early_stopper()

        batch_size = self.optimizer_config.get("batch_size", 32)
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(bundle.train_x, dtype=torch.float32),
                torch.tensor(bundle.train_y, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(bundle.val_x, dtype=torch.float32),
                torch.tensor(bundle.val_y, dtype=torch.float32),
            ),
            batch_size=256,
        )

        max_epochs = self.optimizer_config.get("max_epochs", 50)
        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            val_loss = self._eval_epoch(model, val_loader, loss_fn, device)

            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)

            if self.verbose:
                print(
                    f"Epoch [{epoch}/{max_epochs}] "
                    f"train={train_loss:.6f} val={val_loss:.6f}"
                )

            early_stopper.step(val_loss, model)
            if early_stopper.should_stop:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        early_stopper.restore(model)

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model.predict(bundle.test_x)

        return self._build_result(bundle, preds)

    # ── Sklearn path ────────────────────────────────────────────────

    def _fit_sklearn(self, bundle):
        self.model.fit(
            bundle.train_x, bundle.train_y,
            val_x=bundle.val_x, val_y=bundle.val_y,
        )
        preds = self.model.predict(bundle.test_x)
        return self._build_result(bundle, preds)

    # ── TimeSeries path ─────────────────────────────────────────────

    def _fit_ts(self, bundle):
        y_train = bundle.get_y_series("train").squeeze()
        y_test = bundle.get_y_series("test").squeeze()

        # Strip feature columns that overlap with targets.
        # ARIMA/ETS already model y from its own past, so passing
        # future y as an exogenous variable would leak the answer.
        # Non-overlapping columns (e.g. temp, humidity) are kept as
        # legitimate exogenous variables.
        _overlap = set(bundle.feature_names) & set(bundle.target_names)
        if _overlap:
            _keep = [c for c in bundle.feature_names if c not in _overlap]
            if _keep:
                _idx = [bundle.feature_names.index(c) for c in _keep]
                X_train = bundle.get_X_series("train")[:, _idx]
                X_test = bundle.get_X_series("test")[:, _idx]
            else:
                X_train = X_test = None
        else:
            X_train = bundle.get_X_series("train") if bundle.feature_names else None
            X_test = bundle.get_X_series("test") if bundle.feature_names else None

        # Try with X features first, fall back to univariate
        try:
            self.model.fit_sequence(y_train, X_train)
            preds = self.model.predict_sequence(y_test, X_test, update_state=True)
        except (ValueError, TypeError) as e:
            if "X" in str(e) or "exogenous" in str(e) or "X_train" in str(e):
                self.model.fit_sequence(y_train, None)
                preds = self.model.predict_sequence(y_test, None, update_state=True)
            else:
                raise

        # Ensure preds is numpy array, (N, horizon, target_dim)
        if isinstance(preds, pd.DataFrame):
            preds = preds["y_pred"].values
        preds = np.asarray(preds, dtype=np.float32)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1, 1)
        elif preds.ndim == 2:
            preds = preds.reshape(preds.shape[0], -1, 1)

        return self._build_result(bundle, preds)

    # ── Internal helpers ────────────────────────────────────────────

    def _build_result(self, bundle, predictions) -> TrainResult:
        n = predictions.shape[0]
        y_true = bundle.get_y_series("test")[:n]

        # Simple per-column inverse when transforms exist
        if bundle.transforms is not None and hasattr(bundle.transforms, "inverse"):
            try:
                for i, tn in enumerate(bundle.target_names):
                    col_pred = f"{tn}_pred"
                    inv_df = pd.DataFrame(
                        np.column_stack([y_true[:, i], predictions[:, -1, i]]),
                        columns=[tn, col_pred],
                    )
                    inv_df = bundle.transforms.inverse(inv_df)
                    predictions[:, -1, i] = inv_df[col_pred].values
                    y_true[:, i] = inv_df[tn].values
            except Exception:
                pass  # fall back to raw predictions

        metrics = self._compute_metrics(y_true, predictions, bundle.target_names)
        return TrainResult(
            model=self.model,
            predictions=predictions,
            metrics=metrics,
            history=getattr(self, "_history", None),
        )

    def _compute_metrics(self, y_true, y_pred, target_names) -> pd.DataFrame:
        # y_true: (N, target_dim) or (N,)
        # y_pred: (N, horizon, target_dim) or (N, horizon)
        rows = []
        for i, tgt in enumerate(target_names):
            if y_pred.ndim == 3:
                yp = y_pred[:, :, i].ravel()  # (N * horizon,)
            else:
                yp = y_pred.ravel()

            if y_true.ndim == 2:
                yt = y_true[:, i]
            else:
                yt = y_true

            # Repeat y_true to match horizon-expanded y_pred
            if len(yp) > len(yt) and len(yp) % len(yt) == 0:
                yt = np.tile(yt, len(yp) // len(yt))

            # Truncate to same length
            min_len = min(len(yp), len(yt))
            yp, yt = yp[:min_len], yt[:min_len]

            error = yp - yt
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            nonzero = yt != 0
            mape = float(np.mean(np.abs(error[nonzero] / yt[nonzero]))) * 100 if nonzero.sum() > 0 else np.nan
            ss_res = np.sum(error ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
            corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) >= 2 and np.std(yt) > 0 and np.std(yp) > 0 else np.nan

            rows.append({"target": tgt, "MAE": mae, "RMSE": rmse,
                         "MAPE": mape, "R2": r2, "PearsonR": corr, "n": len(yt)})
        return pd.DataFrame(rows)

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    def _make_early_stopper(self):
        from .train import EarlyStopping
        return EarlyStopping(**self.early_stopping_config)

    def _train_epoch(self, model, loader, loss_fn, optimizer, device):
        import torch
        model.train()
        total_loss, total = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            if self.optimizer_config.get("grad_clip_val"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.optimizer_config["grad_clip_val"]
                )
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        return total_loss / max(total, 1)

    def _eval_epoch(self, model, loader, loss_fn, device):
        model.eval()
        import torch
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                total_loss += loss.item() * x.size(0)
                total += x.size(0)
        return total_loss / max(total, 1)


__all__ = ["EpiAITrainer", "TrainResult"]
