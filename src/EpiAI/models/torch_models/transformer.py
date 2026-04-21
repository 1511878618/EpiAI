import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    """
    基于 Transformer Encoder 的时间序列预测模型

    功能：
    - 输入过去 lookback 个时间步
    - 预测未来 horizon 个时间步
    - 使用自注意力建模长依赖关系

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
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim
        self.d_model = d_model

        # ===== 输入投影 =====
        self.input_proj = nn.Linear(input_dim, d_model)

        # ===== 位置编码（可学习）=====
        self.pos_embedding = nn.Parameter(
            torch.randn(1, lookback, d_model)
        )

        # ===== Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, T, D)
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ===== 输出层 =====
        self.fc = nn.Linear(d_model, horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, lookback, input_dim)
        y: (B, horizon, target_dim)
        """

        bsz = x.shape[0]

        # ===== 输入 embedding =====
        x = self.input_proj(x)                 # (B, T, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # ===== Transformer Encoder =====
        x = self.encoder(x)                   # (B, T, d_model)

        # ===== 取最后时间步 =====
        x = x[:, -1, :]                       # (B, d_model)

        # ===== 映射到未来 =====
        y = self.fc(x)                        # (B, horizon * target_dim)
        y = y.reshape(bsz, self.horizon, self.target_dim)

        return y

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def initialize(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()




