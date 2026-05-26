import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm 

# =========================
# 1. Device
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =========================
# 2. EarlyStopping
# =========================
class EarlyStopping:
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        assert mode in ["min", "max"]
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.best_state_dict = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

    def step(self, current_score: float, model: nn.Module):
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            return

        if self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model: nn.Module):
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


# =========================
# 3. Train config
# =========================
@dataclass
class TrainConfig:
    max_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip_val: Optional[float] = 1.0
    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    use_scheduler: bool = False
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    save_best_path: Optional[str] = "best_model.pt"
    print_every_epoch: bool = True


# =========================
# 4. One epoch: train
# =========================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_val: Optional[float] = None,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # 这里假设 dataloader 返回 (x, y)
        # 如果你的是别的格式，我下面会告诉你怎么改
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()

        if grad_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


# =========================
# 5. One epoch: validate
# =========================
@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


# =========================
# 6. Main fit function
# =========================
def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    config: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    device = get_device()
    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.monitor_mode,
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
        )

    early_stopper = EarlyStopping(
        monitor=config.monitor,
        mode=config.monitor_mode,
        patience=config.patience,
        min_delta=config.min_delta,
        restore_best_weights=True,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": math.inf if config.monitor_mode == "min" else -math.inf,
    }

    print(f"Using device: {device}")

    for epoch in tqdm(range(1, (config.max_epochs + 1))):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_val=config.grad_clip_val,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if config.print_every_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{config.max_epochs}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"lr={current_lr:.6e}"
            )

        # scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # save best
        improved = False
        if config.monitor_mode == "min":
            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                improved = True
        else:
            if val_loss > history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                improved = True

        if improved and config.save_best_path is not None:
            os.makedirs(os.path.dirname(config.save_best_path) or ".", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                config.save_best_path,
            )

        # early stopping
        early_stopper.step(val_loss, model)
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # restore best weights in memory
    early_stopper.restore(model)

    print(f"Best val_loss: {early_stopper.best_score:.6f}")
    return model, history

