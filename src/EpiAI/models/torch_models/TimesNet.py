"""
    x: [B, T, C]
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
try:
    from EpiAI.layers.Embed import DataEmbedding
except ImportError:
    pass
try:
    from EpiAI.layers.Conv_Blocks import Inception_Block_V1
except ImportError:
    pass

from EpiAI.models.base import TorchMixin
from EpiAI.models.registry import register

class TimesBlock(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.k = top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, lookback + horizon, d_model)
        """
        B, T, N = x.size()
        total_len = self.lookback + self.horizon

        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = int(period_list[i])

            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros(
                    (B, length - total_len, N),
                    dtype=x.dtype,
                    device=x.device,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)  # (B, total_len, N, k)

        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, res.shape[1], N, 1)

        res = torch.sum(res * period_weight, dim=-1)
        return res + x


from EpiAI.models.base import TorchMixin
from EpiAI.models.registry import register

@register("TimesNet", "timesnet")
class TimesNetForecaster(nn.Module, TorchMixin):
    """
    基于 TimesNet 的时间序列预测模型

    功能：
    - 输入过去 lookback 个时间步
    - 预测未来 horizon 个时间步
    - 使用频域周期发现 + TimesBlock 提取时序模式

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
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        embed: str = "fixed",
        freq: str = "h",
        output_mode: str = "regression",  # regression | nonnegative | probability
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim
        self.output_mode = output_mode
        self.use_norm = use_norm
        self.layer = e_layers

        self.enc_embedding = DataEmbedding(input_dim, d_model, embed, freq, dropout)

        # 时间维扩展: lookback -> lookback + horizon
        self.predict_linear = nn.Linear(lookback, lookback + horizon)

        self.backbone = nn.ModuleList([
            TimesBlock(
                lookback=lookback,
                horizon=horizon,
                top_k=top_k,
                d_model=d_model,
                d_ff=d_ff,
                num_kernels=num_kernels,
            )
            for _ in range(e_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

        # 逐时间步预测未来 horizon
        self.projection = nn.Linear(d_model, target_dim, bias=True)

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
            raise ValueError(f"Expected lookback={self.lookback}, but got x.shape[1]={x.shape[1]}")

        if x.shape[2] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, but got x.shape[2]={x.shape[2]}")

        # ===== 可选归一化 =====
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
        else:
            x_norm = x
            means = None
            stdev = None

        # ===== Embedding =====
        enc_out = self.enc_embedding(x_norm, None)  # (B, lookback, d_model)

        # ===== 时间维扩展 =====
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # (B, lookback + horizon, d_model)

        # ===== TimesNet 主体 =====
        for i in range(self.layer):
            enc_out = self.layer_norm(self.backbone[i](enc_out))

        # ===== 只保留未来 horizon 段 =====
        future_feat = enc_out[:, -self.horizon:, :]  # (B, horizon, d_model)

        # ===== 逐步预测 =====
        y = self.projection(future_feat)  # (B, horizon, target_dim)

        # ===== 可选反归一化 =====
        # 仅在 target_dim == input_dim 时尝试按输入尺度还原
        if self.use_norm and self.output_mode in {"regression", "nonnegative"}:
            if self.target_dim == self.input_dim:
                y = y * stdev[:, :1, :self.target_dim].repeat(1, self.horizon, 1)
                y = y + means[:, :1, :self.target_dim].repeat(1, self.horizon, 1)

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
