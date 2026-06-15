"""
Lstm
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

@register("LSTM", "lstm")
class LSTMForecaster(nn.Module, TorchMixin):
    """
    基于 LSTM 的多变量时间序列预测模型
    
    该模型使用 LSTM 编码器处理历史序列，并通过全连接层解码生成未来多个时间步的预测。
    适用于多输入多输出的时间序列预测任务。
    
    Args:
        input_dim (int): 每个时间步的输入特征维度
        lookback (int): 历史窗口长度，即使用过去多少个时间步作为输入
        horizon (int): 预测窗口长度，即预测未来多少个时间步
        target_dim (int): 每个未来时间步需要预测的变量数量
        hidden_dim (int, optional): LSTM 隐状态维度，默认为 128
        dropout (float, optional): Dropout 概率，用于防止过拟合，默认为 0.1
        num_layers (int, optional): LSTM 层数，大于 1 时启用内部 dropout，默认为 1
    """

    def __init__(
        self,
        input_dim: int,  # 每个时间步的输入特征维度（例如多个传感器）
        lookback: int,  # 使用过去多少个时间步作为输入
        horizon: int,  # 预测未来多少个时间步
        target_dim: int,  # 每个未来时间步需要预测的变量数量
        hidden_dim: int = 128,  # LSTM 隐状态维度
        dropout: float = 0.1,  # Dropout 概率（用于防止过拟合）
        num_layers: int = 1,  # LSTM 层数（>1 时才启用内部 dropout）
    ) -> None:

        super().__init__()

        # ===== 保存结构参数（方便后续使用）=====
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        # ===== LSTM 编码器 =====
        # 输入:  (B, lookback, input_dim)
        # 输出:  (B, lookback, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,   # 输入输出格式为 (B, T, D)
            dropout=dropout if num_layers > 1 else 0.0,  # 单层LSTM不使用dropout
        )

        # 额外 dropout（作用于最后的时间步特征）
        self.dropout = nn.Dropout(dropout)

        # ===== 全连接层（解码器）=====
        # 将最后一个时间步的 hidden state 映射到未来 horizon * target_dim
        self.fc = nn.Linear(hidden_dim, horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
        - x: 输入序列，shape = (B, lookback, input_dim)

        返回：
        - y: 预测结果，shape = (B, horizon, target_dim)
        """

        # batch size
        bsz = x.shape[0]

        # ===== LSTM 编码 =====
        # out: (B, lookback, hidden_dim)
        # 这里只取输出，不使用 (h_n, c_n)
        out, _ = self.lstm(x)

        # ===== 取最后一个时间步的表示 =====
        # 表示整个历史窗口的信息汇总
        # (B, hidden_dim)
        out = out[:, -1, :]

        # ===== Dropout 防止过拟合 =====
        out = self.dropout(out)

        # ===== 线性映射到未来序列 =====
        # (B, horizon * target_dim)
        y = self.fc(out)

        # ===== reshape 成时间序列形式 =====
        # (B, horizon, target_dim)
        y = y.reshape(bsz, self.horizon, self.target_dim)

        return y
