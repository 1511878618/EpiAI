"""
Temporal Convolutional Network (TCN) for time series forecasting.
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

class TemporalBlock(nn.Module):
    """
    Residual Block for Temporal Convolutional Network (TCN).
    
    This block implements two layers of dilated causal convolutions with 
    residual connections. It ensures the causal constraint by padding 
    and slicing, meaning output[t] only depends on input[:t+1].
    
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int): Convolutional kernel size.
        dilation (int): Dilation rate for dilated convolution.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = torch.relu(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]

        return torch.relu(out + self.downsample(x))


from EpiAI.models.base import TorchMixin
from EpiAI.models.registry import register

@register("TCN", "tcn")
class TCNForecaster(nn.Module, TorchMixin):
    """
    Temporal Convolutional Network (TCN) for Time Series Forecasting.

    Architecture:
        1. Dilated Causal Convolutions: Captures long-range dependencies 
           without recurrent connections or attention mechanisms.
        2. Exponential Dilation: Receptive field grows exponentially 
           with the depth of the network.
        3. Residual Connections: Stabilizes training for deep architectures.

    Input: 
        x: (Batch, Lookback, Input_dim)
    Output: 
        y: (Batch, Horizon, Target_dim)

    Attributes:
        input_dim (int): Number of input features per time step.
        lookback (int): Number of historical time steps used as input.
        horizon (int): Number of future time steps to predict.
        target_dim (int): Number of target features to predict.
        channels (list): List of hidden channel sizes (defines network depth).
        kernel_size (int): Size of the convolutional filters.
    """
    def __init__(
        self,
        input_dim,
        lookback,
        horizon,
        target_dim,
        channels=[32, 32],
        kernel_size=3,
    ):
        super().__init__()

        layers = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_ch, ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i
                )
            )
            in_ch = ch

        self.network = nn.Sequential(*layers)

        self.fc = nn.Linear(channels[-1], horizon * target_dim)
        # ===== 基本结构参数 =====
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, x):
        B = x.shape[0]

        x = x.permute(0, 2, 1)  # (B,C,T)
        x = self.network(x)

        x = x[:, :, -1]  # last time

        y = self.fc(x)
        y = y.reshape(B, self.horizon, self.target_dim)

        return y
