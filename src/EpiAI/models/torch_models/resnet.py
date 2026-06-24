"""
ResNet-based time series forecasting model (ResNet1D).
"""

from __future__ import annotations


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    class _MockModule:
        class Module:
            pass
        class Linear:
            pass
        class Dropout:
            pass
        class ModuleList:
            pass
        class Identity:
            pass
        ReLU = Gelu = Sigmoid = Softplus = Tanh = Identity
        BatchNorm1d = LayerNorm = Identity
        Sequential = Identity
        class Parameter:
            pass
        class init:
            @staticmethod
            def xavier_uniform_(x): return x
            kaiming_uniform_ = zeros_ = ones_ = normal_ = xavier_uniform_
        class functional:
            @staticmethod
            def relu(x): return x
        functional.relu = staticmethod(lambda x: x)
    nn = _MockModule
    class _MockF:
        @staticmethod
        def relu(x): return x
    F = _MockF
from EpiAI.models.base import TorchMixin
from EpiAI.models.registry import register

class ResidualBlock1D(nn.Module):
    """
    1D ResNet 基本残差块

    输入输出:
    - 输入:  (B, C, T)
    - 输出:  (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # shortcut 分支
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + residual
        out = F.relu(out)

        return out


@register("ResNet")
class ResNetForecaster(nn.Module, TorchMixin):
    """
    基于 ResNet 的时间序列预测模型（ResNet1D）

    功能：
    - 输入过去 lookback 个时间步
    - 预测未来 horizon 个时间步
    - 使用残差卷积提取多尺度时间特征

    输入输出：
    - 输入:  (B, lookback, input_dim)
    - 输出:  (B, horizon, target_dim)
    """

    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        target_dim: int,
        base_channels: int = 32,
        num_blocks: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        self.input_norm = nn.LayerNorm(input_dim) if input_dim > 1 else nn.Identity()

        # ===== 输入投影 =====
        self.input_proj = nn.Conv1d(
            in_channels=input_dim,
            out_channels=base_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # ===== ResNet 主体 =====
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock1D(
                    base_channels,
                    base_channels,
                    kernel_size=kernel_size
                )
            )
        self.resnet = nn.Sequential(*blocks)

        # ===== 全局池化 =====
        # 不用 AdaptiveAvgPool1d（会丢失时间信息）
        # 改用 flatten 保留完整时间特征

        # ===== 输出层 =====
        self.fc = nn.Linear(base_channels * lookback, horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, lookback, input_dim)
        y: (B, horizon, target_dim)
        """

        bsz = x.shape[0]

        # (B, T, C) -> (B, C, T)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)

        # 输入投影
        x = self.input_proj(x)

        # ResNet blocks
        x = self.resnet(x)

        # 展平时间+通道维 → FC 输出
        x = x.reshape(bsz, -1)               # (B, C * T)
        y = self.fc(x)                       # (B, horizon * target_dim)
        y = y.reshape(bsz, self.horizon, self.target_dim)

        return y

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def initialize(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
