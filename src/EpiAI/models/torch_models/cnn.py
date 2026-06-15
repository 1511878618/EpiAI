"""
1D CNN-based time series forecasting model.
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

@register("CNN", "cnn")
class CNNForecaster(nn.Module, TorchMixin):
    """
    基于 1D CNN 的时间序列预测模型

    功能：
    - 输入过去 lookback 个时间步
    - 预测未来 horizon 个时间步
    - 使用卷积提取时间局部特征，再通过全连接层完成预测

    输入输出形式：
    - 输入:  (B, lookback, input_dim)
    - 输出:  (B, horizon, target_dim)
    """

    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        target_dim: int,
        dropout: float = 0.5,
        conv1_config: dict = {'hid': 16, 'kernel': 3, 'stride': 1, 'padding': 1},
        conv2_config: dict = {'hid': 32, 'kernel': 3, 'stride': 1, 'padding': 1},
        pool_config: dict = {'kernel': 2, 'stride': 2, 'padding': 0},
        linear_hid: int = 128,
    ) -> None:
        super().__init__()

        # ===== 基本参数 =====
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        # ===== 卷积特征提取 =====
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv1_config['hid'],
            kernel_size=conv1_config['kernel'],
            stride=conv1_config['stride'],
            padding=conv1_config['padding'],
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_config['hid'],
            out_channels=conv2_config['hid'],
            kernel_size=conv2_config['kernel'],
            stride=conv2_config['stride'],
            padding=conv2_config['padding'],
        )

        self.pool = nn.MaxPool1d(
            kernel_size=pool_config['kernel'],
            stride=pool_config['stride'],
            padding=pool_config['padding'],
        )

        # ===== 计算卷积后的时间长度 =====
        # 两次 pool，每次 /2
        conv_out_len = lookback // 4

        # ===== 全连接预测层 =====
        self.fc1 = nn.Linear(conv2_config['hid'] * conv_out_len, linear_hid)
        self.fc2 = nn.Linear(linear_hid, horizon * target_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
        - x: 输入序列，shape = (B, lookback, input_dim)

        返回：
        - y: 预测结果，shape = (B, horizon, target_dim)
        """

        bsz = x.shape[0]

        # ===== CNN 特征提取 =====
        # (B, lookback, input_dim) -> (B, input_dim, lookback)
        x = x.permute(0, 2, 1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # ===== 展平 =====
        x = x.reshape(bsz, -1)

        # ===== 全连接预测 =====
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)  # (B, horizon * target_dim)

        # ===== reshape 输出 =====
        y = y.reshape(bsz, self.horizon, self.target_dim)

        return y

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def initialize(self) -> None:
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
