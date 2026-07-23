"""EpiAI — End-to-end infectious disease outbreak forecasting framework.

Core components
---------------
- ForecastPipeline / PipelineBundle — data pipeline
- EpiAITrainer / TrainResult — model training
- InferencePipeline / ModelVault / DeploymentRuntime — inference & deployment
- ForecastTask — one-shot compare + deploy workflow

Model registry
--------------
>>> from EpiAI.models import get, list_models
>>> model_cls = get("sf_auto_ets")
>>> list_models("ts", exogenous=True)
["sf_auto_arima", "sf_arima", ...]
"""

from EpiAI.dataset import ForecastPipeline, PipelineBundle
from EpiAI.trainer import EpiAITrainer, TrainResult
from EpiAI.inference import InferencePipeline, ModelVault, DeploymentRuntime
from EpiAI.task import ForecastTask
from EpiAI.models.registry import get, list_models

__all__ = [
    "ForecastPipeline",
    "PipelineBundle",
    "EpiAITrainer",
    "TrainResult",
    "InferencePipeline",
    "ModelVault",
    "DeploymentRuntime",
    "ForecastTask",
    "get",
    "list_models",
]
