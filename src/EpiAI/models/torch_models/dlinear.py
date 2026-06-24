"""
Dlinear
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

class MovingAvg(nn.Module):
    """
    时间序列滑动平均模块

    功能：
    - 对输入序列做滑动平均
    - 提取时间序列中的平滑趋势项（trend component）

    输入输出形式：
    - 输入:  (B, T, C)
    - 输出:  (B, T, C)

    说明：
    - 这里使用两端复制 padding 的方式，尽量保持输出序列长度与输入一致
    - AvgPool1d 实际作用在时间维度上
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
        - x: 输入时间序列，shape = (B, T, C)

        返回：
        - moving_mean: 滑动平均后的序列，shape = (B, T, C)
        """

        # 在时间维度两端进行复制填充
        # 以保证滑动平均后序列长度基本不变
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)  # (B, T + 2*pad_len, C)

        # AvgPool1d 的输入格式要求为 (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # 回到 (B, T, C)

        return x


class SeriesDecomp(nn.Module):
    """
    时间序列分解模块

    功能：
    - 将输入序列分解为：
      1. seasonal / residual（残差或波动项）
      2. trend（趋势项）

    分解方式：
    - trend = moving average(x)
    - seasonal = x - trend

    输入输出形式：
    - 输入:  (B, T, C)
    - 输出:
        seasonal: (B, T, C)
        trend:    (B, T, C)
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
        - x: 输入时间序列，shape = (B, T, C)

        返回：
        - seasonal: 残差/季节项，shape = (B, T, C)
        - trend: 趋势项，shape = (B, T, C)
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


@register("DLinear")
class DLinearForecaster(nn.Module, TorchMixin):
    """
    基于序列分解的线性预测模型（DLinear 风格）

    功能：
    - 输入过去 lookback 个时间步的数据
    - 预测未来 horizon 个时间步
    - 先将输入序列分解为 seasonal 和 trend 两部分
    - 再分别进行线性映射，最后相加得到预测结果

    当前实现特点：
    - 输入支持多变量（input_dim）
    - 当前输出限制为单目标预测（target_dim=1）
    - 输出形式统一为 (B, horizon, target_dim)，便于整合进预测框架

    输入输出形式：
    - 输入:  (B, lookback, input_dim)
    - 输出:  (B, horizon, 1)

    说明：
    - 该实现中 seasonal 和 trend 共用同一个线性映射结构
    - 模型最终会对特征维做平均，得到单目标输出
    """

    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        target_dim: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        # ===== 基本结构参数 =====
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        # 当前版本只支持单目标预测
        if target_dim != 1:
            raise ValueError(
                f"{type(self).__name__} currently supports only univariate forecasting "
                f"(target_dim=1), but got target_dim={target_dim}."
            )

        # ===== 序列分解模块 =====
        self.decomposition = SeriesDecomp(kernel_size=kernel_size)

        # ===== 线性映射层 =====
        # 将 lookback × input_dim 展平后映射到 horizon × input_dim
        # seasonal 和 trend 分支共用相同结构，但前向中分别调用
        self.linear = nn.Linear(
            in_features=lookback * input_dim,
            out_features=horizon * input_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
        - x: 输入序列，shape = (B, lookback, input_dim)

        返回：
        - y: 预测结果，shape = (B, horizon, 1)

        计算流程：
        1. 对输入序列进行分解，得到 seasonal 和 trend
        2. 分别展平后送入线性层
        3. reshape 回未来时间序列形式
        4. 将两个分支结果相加
        5. 对特征维求均值，得到单目标预测
        """

        # batch size
        bsz = x.shape[0]

        # ===== 序列分解 =====
        # seasonal_init: (B, lookback, input_dim)
        # trend_init:    (B, lookback, input_dim)
        seasonal_init, trend_init = self.decomposition(x)

        # ===== 展平输入 =====
        # (B, lookback * input_dim)
        seasonal_flat = seasonal_init.reshape(bsz, -1)
        trend_flat = trend_init.reshape(bsz, -1)

        # ===== 分别进行线性映射 =====
        # (B, horizon * input_dim)
        seasonal_out = self.linear(seasonal_flat)
        trend_out = self.linear(trend_flat)

        # ===== reshape 回时间序列形式 =====
        # (B, horizon, input_dim)
        seasonal_out = seasonal_out.reshape(bsz, self.horizon, self.input_dim)
        trend_out = trend_out.reshape(bsz, self.horizon, self.input_dim)

        # ===== 合并 seasonal 和 trend 两个分支 =====
        # (B, horizon, input_dim)
        y = seasonal_out + trend_out

        # ===== 压缩特征维，得到单目标预测 =====
        # (B, horizon)
        y = y.mean(dim=2)

        # 保持输出维度统一为 (B, horizon, target_dim)
        # (B, horizon, 1)
        y = y.unsqueeze(-1)

        return y

    def reset_parameters(self) -> None:
        """
        重置模型参数

        说明：
        - 当前需要显式重置的主要是线性层参数
        - 分解模块中没有可学习参数，因此无需额外处理
        """
        self.linear.reset_parameters()

    def initialize(self) -> None:
        """
        初始化模型参数

        说明：
        - 遍历所有子模块
        - 若模块实现了 reset_parameters，则自动调用
        - 便于统一纳入训练框架初始化流程
        """
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
