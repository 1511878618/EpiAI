import torch
import torch.nn as nn
class MLPForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        target_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        self.net = nn.Sequential(
            nn.Linear(lookback * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, lookback, input_dim)
        bsz = x.shape[0]
        x = x.reshape(bsz, -1)
        y = self.net(x)
        y = y.reshape(bsz, self.horizon, self.target_dim)
        return y