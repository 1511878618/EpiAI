import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMForecaster(nn.Module):
    """
    CNN + LSTM 时间序列预测模型

    功能：
    - CNN 提取局部时间模式（短期模式）
    - LSTM 建模时序依赖（长期关系）
    - 输入 lookback → 输出 horizon

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
        cnn_channels: int = 32,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        # ===== CNN 特征提取 =====
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.bn = nn.BatchNorm1d(cnn_channels)

        # ===== LSTM 时序建模 =====
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # ===== 输出层 =====
        self.fc = nn.Linear(lstm_hidden, horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, lookback, input_dim)
        y: (B, horizon, target_dim)
        """

        bsz = x.shape[0]

        # ===== CNN 提取局部特征 =====
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn(self.conv(x)))   # (B, cnn_channels, T)

        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        # ===== LSTM 建模时序 =====
        out, _ = self.lstm(x)              # (B, T, lstm_hidden)

        # 取最后时间步
        out = out[:, -1, :]                # (B, lstm_hidden)

        out = self.dropout(out)

        # ===== 预测未来 =====
        y = self.fc(out)                  # (B, horizon * target_dim)
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
