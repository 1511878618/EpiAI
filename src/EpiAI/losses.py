import torch
import torch.nn as nn
import torch.nn.functional as F

def divide_no_nan(a, b):
    result = a / b
    result[torch.isnan(result) | torch.isinf(result)] = 0.0
    return result

class MAPELoss(nn.Module):
    """均平均绝对百分比误差：衡量相对误差，对大数值不敏感"""
    def forward(self, pred, target):
        # 避免除以 0
        loss = torch.abs((pred - target) / (target + 1e-8))
        return torch.mean(loss)

class SMAPELoss(nn.Module):
    """对称平均绝对百分比误差：修正了 MAPE 对预测值和真实值不对称的问题"""
    def forward(self, pred, target):
        numerator = torch.abs(pred - target)
        denominator = (torch.abs(pred) + torch.abs(target)) / 2 + 1e-8
        return torch.mean(numerator / denominator)


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target + 1e-12))
        return torch.mean(loss)    

class CorrelationLoss(nn.Module):
    """
    相关性损失：关注波形的一致性而非绝对数值。
    适合短期内关注趋势正负（涨跌）而非绝对大小的任务。
    """
    def forward(self, pred, target):
        # 计算 Pearson 相关系数
        pred_mean = torch.mean(pred, dim=1, keepdim=True)
        target_mean = torch.mean(target, dim=1, keepdim=True)
        
        nom = torch.sum((pred - pred_mean) * (target - target_mean), dim=1)
        den = torch.sqrt(torch.sum((pred - pred_mean)**2, dim=1) * torch.sum((target - target_mean)**2, dim=1) + 1e-8)
        
        corr = nom / den
        return 1 - torch.mean(corr)
    
class MultiQuantileLoss(nn.Module):
    """针对 0.1, 0.5, 0.9 分位数的联合损失"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        # 假设 pred 的最后一个维度大小等于 len(quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.stack(losses))
    


class TrendAwareLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        mae_loss = torch.abs(y_pred - y_true).mean()

        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]
        trend_loss = torch.abs(pred_diff - true_diff).mean()

        return self.alpha * mae_loss + self.beta * trend_loss
import torch
import torch.nn as nn

class OutbreakAwareLoss(nn.Module):
    def __init__(
        self,
        outbreak_threshold,
        alpha=1.0,          # base MAE 权重
        beta=3.0,           # outbreak loss 权重
        gamma=1.0,          # trend loss 权重
        outbreak_weight=5.0 # 爆发点额外权重
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.outbreak_weight = outbreak_weight

    def forward(self, pred, target):
        """
        pred:   [B, H, D]
        target: [B, H, D]
        """
        # 1) 基础 MAE
        base_mae = torch.abs(pred - target).mean()

        # 2) 爆发窗口损失
        outbreak_mask = (target >= self.outbreak_threshold).float()   # [B, H, D]

        # 给爆发点更大权重
        weights = 1.0 + outbreak_mask * (self.outbreak_weight - 1.0)
        outbreak_mae = (weights * torch.abs(pred - target)).mean()

        # 3) 趋势损失（比较 horizon 内的变化）
        if target.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            trend_mae = torch.abs(pred_diff - target_diff).mean()
        else:
            trend_mae = torch.tensor(0.0, device=target.device)

        loss = self.alpha * base_mae + self.beta * outbreak_mae + self.gamma * trend_mae
        return loss

class AsymmetricOutbreakLoss(nn.Module):
    def __init__(self, outbreak_threshold, alpha=1.0, beta=3.0, gamma=1.0, under_weight=2.0):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.under_weight = under_weight

    def forward(self, pred, target):
        base_mae = torch.abs(pred - target).mean()

        outbreak_mask = (target >= self.outbreak_threshold).float()
        outbreak_count = outbreak_mask.sum()

        if outbreak_count > 0:
            error = pred - target
            abs_error = torch.abs(error)

            # 低估：pred < target
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutbreakWeightedHuberLoss(nn.Module):
    """
    pred:   [B, H, D]
    target: [B, H, D]

    组成：
    1) base huber：整体稳定
    2) outbreak window mae：强化爆发窗口
    3) trend mae：强化窗口内升降趋势
    """
    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,   # base huber
        beta: float = 3.0,    # outbreak window
        gamma: float = 1.0,   # trend
        delta: float = 1.0,   # huber delta
        expand_window: bool = True
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
        mask: [B, H, D]
        若某一步是爆发，则前后一步也纳入爆发窗口
        """
        if not self.expand_window or mask.size(1) <= 1:
            return mask

        expanded = mask.clone()
        expanded[:, 1:, :] = torch.maximum(expanded[:, 1:, :], mask[:, :-1, :])
        expanded[:, :-1, :] = torch.maximum(expanded[:, :-1, :], mask[:, 1:, :])
        return expanded

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"pred {pred.shape} != target {target.shape}"

        # 1) base huber
        base_huber = F.huber_loss(pred, target, delta=self.delta, reduction="mean")

        # 2) outbreak window mae
        raw_mask = (target >= self.outbreak_threshold).float()
        outbreak_mask = self._expand_mask(raw_mask)
        outbreak_count = outbreak_mask.sum()

        if outbreak_count > 0:
            outbreak_mae = (torch.abs(pred - target) * outbreak_mask).sum() / outbreak_count
        else:
            outbreak_mae = pred.new_tensor(0.0)

        # 3) trend mae
        if target.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            trend_mae = torch.abs(pred_diff - target_diff).mean()
        else:
            trend_mae = pred.new_tensor(0.0)

        loss = self.alpha * base_huber + self.beta * outbreak_mae + self.gamma * trend_mae
        return loss

class AsymmetricOutbreakLoss_V2(nn.Module):
    """
    更惩罚“低估爆发”的情况：
    pred < target 且 target 处于爆发窗口时，额外加权
    """
    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,         # base mae
        beta: float = 3.0,          # outbreak part
        gamma: float = 1.0,         # trend
        under_weight: float = 2.0,  # 对低估额外惩罚
        expand_window: bool = True
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

        # 1) base mae
        base_mae = torch.abs(pred - target).mean()

        # 2) outbreak window asymmetric loss
        raw_mask = (target >= self.outbreak_threshold).float()
        outbreak_mask = self._expand_mask(raw_mask)
        outbreak_count = outbreak_mask.sum()

        if outbreak_count > 0:
            error = pred - target
            abs_error = torch.abs(error)

            # 低估时 pred-target < 0
            under_mask = (error < 0).float()

            outbreak_loss = (
                abs_error
                * outbreak_mask
                * (1.0 + under_mask * (self.under_weight - 1.0))
            ).sum() / outbreak_count
        else:
            outbreak_loss = pred.new_tensor(0.0)

        # 3) trend
        if target.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            trend_loss = torch.abs(pred_diff - target_diff).mean()
        else:
            trend_loss = pred.new_tensor(0.0)

        return self.alpha * base_mae + self.beta * outbreak_loss + self.gamma * trend_loss
class FocalRegressionLoss(nn.Module):
    """
    在 MAE 基础上加 focal-style 调制：
    error 越大，权重越大
    同时可叠加 outbreak 权重
    """
    def __init__(
        self,
        outbreak_threshold: float = None,
        gamma_focal: float = 2.0,
        outbreak_weight: float = 3.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.outbreak_threshold = outbreak_threshold
        self.gamma_focal = gamma_focal
        self.outbreak_weight = outbreak_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"pred {pred.shape} != target {target.shape}"

        abs_error = torch.abs(pred - target)

        # 归一化误差，避免数值过大
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
    不修改模型结构：
    - 主任务：回归（MAE / Huber）
    - 辅助任务：用 pred 相对 threshold 的位置，构造 outbreak 概率，再做 BCE

    注意：
    这里的 outbreak_prob 不是单独 head 输出，而是从 pred 经过 sigmoid 近似得到
    """
    def __init__(
        self,
        outbreak_threshold: float,
        alpha: float = 1.0,      # regression
        beta: float = 1.0,       # outbreak BCE
        use_huber: bool = True,
        delta: float = 1.0,
        temperature: float = 1.0
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

        # 1) regression part
        if self.use_huber:
            reg_loss = F.huber_loss(pred, target, delta=self.delta, reduction="mean")
        else:
            reg_loss = torch.abs(pred - target).mean()

        # 2) outbreak proxy classification
        outbreak_label = (target >= self.outbreak_threshold).float()

        # 用 pred 相对 threshold 的大小构造“爆发概率”
        outbreak_logit = (pred - self.outbreak_threshold) / self.temperature
        cls_loss = F.binary_cross_entropy_with_logits(outbreak_logit, outbreak_label)

        return self.alpha * reg_loss + self.beta * cls_loss
