"""EpiAI — End-to-end infectious disease outbreak forecasting framework.

Core components
---------------
- ``ForecastPipeline`` / ``PipelineBundle`` — data pipeline
- ``EpiAITrainer`` / ``TrainResult`` — model training
- ``InferencePipeline`` / ``ModelVault`` / ``DeploymentRuntime`` — inference & deployment

Model registry
--------------
>>> from EpiAI.models.registry import get, list_models
>>> model_cls = get("RF")
>>> model = model_cls(input_dim=8, lookback=12, horizon=3, target_dim=1)
>>> list_models("torch")
["cnn", "lstm", "mlp", ...]
"""

from EpiAI.dataset import ForecastPipeline, PipelineBundle
from EpiAI.trainer import EpiAITrainer, TrainResult
from EpiAI.inference import InferencePipeline, ModelVault, DeploymentRuntime

__all__ = [
    "ForecastPipeline",
    "PipelineBundle",
    "EpiAITrainer",
    "TrainResult",
    "InferencePipeline",
    "ModelVault",
    "DeploymentRuntime",
]
