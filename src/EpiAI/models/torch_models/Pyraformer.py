import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from EpiAI.layers.Pyraformer_EncDec import Encoder


class _PyraformerConfig:
    """
    轻量配置对象，只给 Encoder 用。
    """
    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        activation: str,
        embed: str,
        freq: str,
    ):
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.dropout = dropout
        self.activation = activation
        self.embed = embed
        self.freq = freq


class PyraformerForecaster(nn.Module):
    """
    基于 Pyraformer 的时间序列预测模型

    功能：
    - 输入过去 lookback 个时间步
    - 预测未来 horizon 个时间步
    - 使用金字塔注意力提取多尺度时序依赖

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
        dropout: float = 0.1,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        window_size: list[int] = [4, 4],
        inner_size: int = 5,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_mode: str = "regression",   # regression | nonnegative | probability
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim
        self.output_mode = output_mode
        self.use_norm = use_norm
        self.window_size = window_size

        configs = _PyraformerConfig(
            enc_in=input_dim,
            seq_len=lookback,
            pred_len=horizon,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
            embed=embed,
            freq=freq,
        )

        self.encoder = Encoder(configs, window_size=window_size, inner_size=inner_size)

        # 原版 forecast 的输出头
        self.projection = nn.Linear(
            (len(window_size) + 1) * d_model,
            horizon * target_dim
        )

        self.dropout = nn.Dropout(dropout)

    def _apply_output_constraint(self, y: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "probability":
            return torch.sigmoid(y)
        if self.output_mode == "nonnegative":
            return F.softplus(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
        - x: 输入序列，shape = (B, lookback, input_dim)

        返回：
        - y: 预测结果，shape = (B, horizon, target_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, lookback, input_dim), got {x.shape}")

        if x.shape[1] != self.lookback:
            raise ValueError(
                f"Expected lookback={self.lookback}, but got x.shape[1]={x.shape[1]}"
            )

        if x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got x.shape[2]={x.shape[2]}"
            )

        # ===== 可选归一化 =====
        if self.use_norm:
            mean_x = x.mean(dim=1, keepdim=True).detach()
            x_norm = x - mean_x
            std_x = torch.sqrt(
                torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_norm = x_norm / std_x
        else:
            x_norm = x
            mean_x = None
            std_x = None

        # ===== 编码 =====
        # Encoder 输出通常是 [B, T', hidden]
        enc_out = self.encoder(x_norm, None)

        # ===== 取最后一个聚合状态 =====
        final_state = enc_out[:, -1, :]   # [B, (len(window_size)+1)*d_model]

        # ===== 预测未来 horizon =====
        y = self.projection(self.dropout(final_state))
        y = y.view(x.shape[0], self.horizon, self.target_dim)

        # ===== 可选反归一化 =====
        # 仅在 target_dim == input_dim 时尝试按输入尺度还原
        if self.use_norm and self.output_mode in {"regression", "nonnegative"}:
            if self.target_dim == self.input_dim:
                y = y * std_x[:, :1, :self.target_dim].repeat(1, self.horizon, 1)
                y = y + mean_x[:, :1, :self.target_dim].repeat(1, self.horizon, 1)

        # ===== 输出约束 =====
        y = self._apply_output_constraint(y)
        return y

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def initialize(self) -> None:
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
