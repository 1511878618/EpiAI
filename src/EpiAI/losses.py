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