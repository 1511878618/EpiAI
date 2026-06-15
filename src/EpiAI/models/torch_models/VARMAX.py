
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

