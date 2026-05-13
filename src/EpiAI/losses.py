"""
Loss functions for infectious disease outbreak forecasting.

This module provides various loss functions designed to handle the
imbalanced nature of outbreak prediction, where爆发期 (outbreak periods)
require special attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Safe division that handles NaN and Inf values."""
    result = a / b
    result[torch.isnan(result) | torch.isinf(result)] = 0.0
    return result


# =============================================================================
# Basic Losses
# =============================================================================


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error - measures relative error, insensitive to large values."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.abs((pred - target) / (target + 1e-8))
        return torch.mean(loss)


class SMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error.
    Addresses the asymmetry issue of MAPE by using symmetric denominator.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = torch.abs(pred - target)
        denominator = (torch.abs(pred) + torch.abs(target)) / 2 + 1e-8
        return torch.mean(numerator / denominator)


class LogCoshLoss(nn.Module):
    """Log-Cosh loss function - behaves like MSE but is smoother."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(pred - target + 1e-12))
        return torch.mean(loss)


class CorrelationLoss(nn.Module):
    """
    Correlation-based loss focusing on waveform consistency.
    Suitable for tasks where trend direction matters more than absolute values.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = torch.mean(pred, dim=1, keepdim=True)
        target_mean = torch.mean(target, dim=1, keepdim=True)

        nom = torch.sum((pred - pred_mean) * (target - target_mean), dim=1)
        den = torch.sqrt(
            torch.sum((pred - pred_mean) ** 2, dim=1)
            * torch.sum((target - target_mean) ** 2, dim=1)
            + 1e-8
        )

        corr = nom / den
        return 1 - torch.mean(corr)


class MultiQuantileLoss(nn.Module):
    """Joint loss for multiple quantiles (0.1, 0.5, 0.9)."""

    def __init__(self, quantiles: list[float] | tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = list(quantiles)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.stack(losses))


# =============================================================================
# Trend-Aware Losses
# =============================================================================


class TrendAwareLoss(nn.Module):
    """
    Combines MAE with trend consistency loss.
    Ensures predictions follow the same trend pattern as ground truth.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mae_loss = torch.abs(y_pred - y_true).mean()

        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]
        trend_loss = torch.abs(pred_diff - true_diff).mean()

        return self.alpha * mae_loss + self.beta * trend_loss


# =============================================================================
# Outbreak-Aware Losses
# =============================================================================


class OutbreakAwareLoss(nn.Module):
    """
    Outbreak-aware loss with weighted MAE.
    Gives higher weight to samples during outbreak periods.

    Args:
        outbreak_threshold: Threshold to identify outbreak periods
        alpha: Weight for base MAE
        beta: Weight for outbreak loss
        gamma: Weight for trend loss
        outbreak_weight: Additional weight multiplier for outbreak samples
    """

    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,
        beta: float = 3.0,
        gamma: float = 1.0,
        outbreak_weight: float = 5.0,
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.outbreak_weight = outbreak_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Base MAE
        base_mae = torch.abs(pred - target).mean()

        # Outbreak window loss
        outbreak_mask = (target >= self.outbreak_threshold).float()
        weights = 1.0 + outbreak_mask * (self.outbreak_weight - 1.0)
        outbreak_mae = (weights * torch.abs(pred - target)).mean()

        # Trend loss
        if target.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            trend_mae = torch.abs(pred_diff - target_diff).mean()
        else:
            trend_mae = torch.tensor(0.0, device=target.device)

        return self.alpha * base_mae + self.beta * outbreak_mae + self.gamma * trend_mae


class AsymmetricOutbreakLoss(nn.Module):
    """
    Asymmetric outbreak loss that penalizes underestimation more heavily.
    During outbreak periods, underestimating (pred < target) gets extra penalty.
    """

    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,
        beta: float = 3.0,
        gamma: float = 1.0,
        under_weight: float = 2.0,
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.under_weight = under_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_mae = torch.abs(pred - target).mean()

        outbreak_mask = (target >= self.outbreak_threshold).float()
        outbreak_count = outbreak_mask.sum()

        if outbreak_count > 0:
            error = pred - target
            abs_error = torch.abs(error)
            under_mask = (error < 0).float()

            outbreak_loss = (
                abs_error * outbreak_mask * (1.0 + under_mask * (self.under_weight - 1.0))
            ).sum() / outbreak_count
        else:
            outbreak_loss = torch.tensor(0.0, device=target.device)

        if target.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            trend_loss = torch.abs(pred_diff - target_diff).mean()
        else:
            trend_loss = torch.tensor(0.0, device=target.device)

        return self.alpha * base_mae + self.beta * outbreak_loss + self.gamma * trend_loss


class OutbreakWeightedHuberLoss(nn.Module):
    """
    Combines Huber loss with outbreak awareness and trend consistency.

    Components:
        1. Base Huber loss for overall stability
        2. Outbreak window MAE for强化爆发窗口
        3. Trend MAE for窗口内升降趋势
    """

    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,
        beta: float = 3.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        expand_window: bool = True,
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.expand_window = expand_window

    def _expand_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Expand outbreak mask to include neighboring time steps.

        Args:
            mask: Binary mask of shape [B, H, D]

        Returns:
            Expanded mask where adjacent steps are also marked during outbreaks
        """
        if not self.expand_window or mask.size(1) <= 1:
            return mask

        expanded = mask.clone()
        expanded[:, 1:, :] = torch.maximum(expanded[:, 1:, :], mask[:, :-1, :])
        expanded[:, :-1, :] = torch.maximum(expanded[:, :-1, :], mask[:, 1:, :])
        return expanded

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"pred {pred.shape} != target {target.shape}"

        # Base Huber
        base_huber = F.huber_loss(pred, target, delta=self.delta, reduction="mean")

        # Outbreak window MAE
        raw_mask = (target >= self.outbreak_threshold).float()
        outbreak_mask = self._expand_mask(raw_mask)
        outbreak_count = outbreak_mask.sum()

        if outbreak_count > 0:
            outbreak_mae = (torch.abs(pred - target) * outbreak_mask).sum() / outbreak_count
        else:
            outbreak_mae = pred.new_tensor(0.0)

        # Trend MAE
        if target.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            trend_mae = torch.abs(pred_diff - target_diff).mean()
        else:
            trend_mae = pred.new_tensor(0.0)

        return self.alpha * base_huber + self.beta * outbreak_mae + self.gamma * trend_mae


class AsymmetricOutbreakLossV2(nn.Module):
    """
    V2 of asymmetric outbreak loss with expanded window support.

    Penalizes underestimation (pred < target) more heavily during outbreaks.
    """

    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,
        beta: float = 3.0,
        gamma: float = 1.0,
        under_weight: float = 2.0,
        expand_window: bool = True,
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.under_weight = under_weight
        self.expand_window = expand_window

    def _expand_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.expand_window or mask.size(1) <= 1:
            return mask
        expanded = mask.clone()
        expanded[:, 1:, :] = torch.maximum(expanded[:, 1:, :], mask[:, :-1, :])
        expanded[:, :-1, :] = torch.maximum(expanded[:, :-1, :], mask[:, 1:, :])
        return expanded

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"pred {pred.shape} != target {target.shape}"

        # Base MAE
        base_mae = torch.abs(pred - target).mean()

        # Outbreak window asymmetric loss
        raw_mask = (target >= self.outbreak_threshold).float()
        outbreak_mask = self._expand_mask(raw_mask)
        outbreak_count = outbreak_mask.sum()

        if outbreak_count > 0:
            error = pred - target
            abs_error = torch.abs(error)
            under_mask = (error < 0).float()

            outbreak_loss = (
                abs_error
                * outbreak_mask
                * (1.0 + under_mask * (self.under_weight - 1.0))
            ).sum() / outbreak_count
        else:
            outbreak_loss = pred.new_tensor(0.0)

        # Trend loss
        if target.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            trend_loss = torch.abs(pred_diff - target_diff).mean()
        else:
            trend_loss = pred.new_tensor(0.0)

        return self.alpha * base_mae + self.beta * outbreak_loss + self.gamma * trend_loss


class FocalRegressionLoss(nn.Module):
    """
    Focal-style modulated regression loss.

    Larger errors receive higher weights, with optional outbreak weighting.
    """

    def __init__(
        self,
        outbreak_threshold: float | None = None,
        gamma_focal: float = 2.0,
        outbreak_weight: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.gamma_focal = gamma_focal
        self.outbreak_weight = outbreak_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"pred {pred.shape} != target {target.shape}"

        abs_error = torch.abs(pred - target)

        # Normalize error to avoid numerical issues
        scale = target.abs().mean().detach() + self.eps
        norm_error = abs_error / scale

        focal_factor = (norm_error + self.eps) ** self.gamma_focal

        weights = torch.ones_like(target)
        if self.outbreak_threshold is not None:
            outbreak_mask = (target >= self.outbreak_threshold).float()
            weights = weights + outbreak_mask * (self.outbreak_weight - 1.0)

        loss = weights * focal_factor * abs_error
        return loss.mean()


class RegressionWithOutbreakBCELoss(nn.Module):
    """
    Combined regression and outbreak classification loss.

    - Main task: Regression (MAE / Huber)
    - Auxiliary task: Outbreak probability via BCE on pred relative to threshold

    Note:
        Outbreak probability is derived from pred via sigmoid, not from a separate head.
    """

    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        use_huber: bool = True,
        delta: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.alpha = alpha
        self.beta = beta
        self.use_huber = use_huber
        self.delta = delta
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"pred {pred.shape} != target {target.shape}"

        # Regression part
        if self.use_huber:
            reg_loss = F.huber_loss(pred, target, delta=self.delta, reduction="mean")
        else:
            reg_loss = torch.abs(pred - target).mean()

        # Outbreak proxy classification
        outbreak_label = (target >= self.outbreak_threshold).float()
        outbreak_logit = (pred - self.outbreak_threshold) / self.temperature
        cls_loss = F.binary_cross_entropy_with_logits(outbreak_logit, outbreak_label)

        return self.alpha * reg_loss + self.beta * cls_loss
