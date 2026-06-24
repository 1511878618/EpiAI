"""
Source: https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py
Refactored for EpiAI integration.
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
    from EpiAI.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
except ImportError:
    pass
try:
    from EpiAI.layers.Embed import DataEmbedding_wo_pos
except ImportError:
    pass
try:
    from EpiAI.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
except ImportError:
    pass

from EpiAI.models.base import TorchMixin
from EpiAI.models.registry import register

@register("Autoformer")
class AutoformerForecaster(nn.Module, TorchMixin):
    """
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """
    def __init__(self, 
                 lookback=96,        # 回视窗口长度 (lookback)

                 horizon=96,       # 预测/还原长度 (horizon)
                 input_dim=1,          # 输入特征数
                 target_dim=1,           # 输出特征数
                 d_model=128,       # 隐藏层维度 
                 n_heads=8,         # 多头注意力头数
                 e_layers=2,        # 编码器层数
                 d_ff=256,          # 预测头前馈网络维度
                 moving_avg=25,     # 序列分解的移动平均窗口大小
                 factor=1,          # Auto-Correlation 因子
                 dropout=0.1,       # Dropout 比例
                 embed='fixed',     # 时间特征嵌入方式 [fixed, learned]
                 freq='h',          # 时间频率
                 activation='gelu'):
        super(AutoformerForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.target_dim = target_dim

        # 序列分解算子 (Decomposition)
        self.decomp = series_decomp(moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(input_dim, d_model, embed, freq, dropout)


        # 编码器 (Encoder)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        self.projection = nn.Linear(
                        d_model, horizon*target_dim, bias=True)
            
    
 

    def forward(self, x_enc):
        """
        x_enc: [Batch, seq_len, input_dim]

        """
       # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final: 取编码器最后时间步 → 投影到 horizon 步
        y = self.projection(enc_out[:, -1, :])
        y = y.reshape(x_enc.shape[0], self.horizon, self.target_dim)
        return y 
 