from __future__ import annotations

# Standard library imports
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local application imports
from EpiAI.losses import *
import EpiAI.models.tabular_models as EpiAI_tabular_models
import EpiAI.models.torch_models as EpiAI_torch_models
import EpiAI.models.arima_models as EpiAI_arima_models 

from .train import TrainConfig, fit_model, get_device
from .dataset.time_seris_task_data import SimpleForecastDataModule
from .utils import predict_forecasts
def train_prediction_model(
    data: pd.DataFrame,
    Case_col: Union[str, List[str]],
    Feature_col: Union[str, List[str]],
    Time_col: str,
    model_config: Dict[str, Any],
):
    """
    该任务基于时间序列数据，单一构建模型进行预测
    自动调度器：根据 model_config["modelType"] 将训练任务路由到
    train_torch_model（torch 模型）或 train_tabular_model（XGB/LGBM）

    Parameters
    ----------
    data : pd.DataFrame
        原始数据，需包含 Time_col、Case_col 和 Feature_col 对应的列。
    Case_col : str or list[str]
        目标变量列名（预测目标）。
    Feature_col : str or list[str]
        输入特征列名。
    Time_col : str
        时间列名。
    model_config : dict
        模型配置，键值说明：

        ================== ============ ===========================================
        键名                必填          说明
        ================== ============ ===========================================
        modelType          是            模型类型，如 "LSTM"、"XGB"、"TabPFN"
        params             否            模型参数，含 lookback / horizon / dropout 等
        training_config    否            训练参数，含 max_epochs / lr / patience 等
        ================== ============ ===========================================

        **Torch 模型示例 (modelType="LSTM")** ::

            {
                "modelType": "LSTM",
                "params": {"lookback": 12, "horizon": 3, "dropout": 0.1, "batch_size": 32},
                "training_config": {"max_epochs": 100, "lr": 1e-3, "train_val_test_ratio": (6, 2, 2)},
            }

        **Tabular 模型示例 (modelType="XGBSingle")** ::

            {
                "modelType": "XGBSingle",
                "params": {"lookback": 12, "horizon": 3},
                "training_config": {"train_val_test_ratio": (6, 2, 2), "fillna_method": "zero"},
            }

    Returns
    -------
    dict
        训练输出，因底层模型而异：

        - **Torch 模型**:
            { model, result_df, metrics_df, datamodule }
        - **Tabular 模型**:
            { model, result_df, metrics_df, datamodule }

    Raises
    ------
    ValueError
        - model_config 中缺少 "modelType" 键
        - modelType 对应的模型名不在 EpiAI_torch_models 或 EpiAI_tabular_models 中

    Notes
    -----
    调度规则: modelType → 拼接 "Forecaster" 后缀 → 依次在 torch 和 tabular
    模型库中查找，优先匹配 torch 模型。

    当前支持模型:

        **Torch**: CNNLSTM, CNN, DLinear, LSTM, MLP, ResNet, TCN, Transformer, Autoformer, TimesNet

        **Tabular**: XGB, LGBM, TabPFN

    Example
    -------
    >>> out = train_prediction_model(
    ...     data=df,
    ...     Case_col="AIDS",
    ...     Feature_col=["Influenza", "HFMD"],
    ...     Time_col="Year/Month",
    ...     model_config={
    ...         "modelType": "LSTM",
    ...         "params": {"lookback": 12, "horizon": 3},
    ...         "training_config": {"max_epochs": 50},
    ...     },
    ... )
    >>> out["result_df"].head()
    """

    model_type = model_config.get("modelType", None)
    if model_type is None:
        raise ValueError("model_config['modelType'] is required.")

    modelname = f"{model_type}Forecaster"

    # ---- Torch 模型 ----
    if modelname in EpiAI_torch_models.__dict__:
   
        return train_torch_model(
            data=data,
            Case_col=Case_col,
            Feature_col=Feature_col,
            Time_col=Time_col,
            model_config=model_config,
        )

    # ---- Tabular 模型 ----
    if modelname in EpiAI_tabular_models.__dict__:
        return train_tabular_model(
            data=data,
            Case_col=Case_col,
            Feature_col=Feature_col,
            Time_col=Time_col,
            model_config=model_config,
        )

    if modelname in EpiAI_arima_models.__dict__:
        return train_arima_model(
            data=data,
            Case_col=Case_col,
            Feature_col=Feature_col,
            Time_col=Time_col,
            model_config=model_config,
        )

    # ---- 都不匹配 ----
    raise ValueError(
        f"Model {modelname!r} not found. "
        f"Available torch models: {list(EpiAI_torch_models.__dict__.keys())} | "
        f"tabular models: {list(EpiAI_tabular_models.__dict__.keys())} | "
        f"ARIMA models:{list(EpiAI_arima_models.__dict__.keys())} "
    )


def _evaluate_prediction_metrics(
    df: pd.DataFrame,
    target_feature_names: list[str],
) -> pd.DataFrame:
    """
    根据 result_df 里的真实值和 prediction_xxx 计算指标。

    计算：
        MAE
        RMSE
        MAPE
        R2          (新增)
        PearsonR    (新增)

    会分别计算：
        All
        Train
        Val
        Test
    """

    rows = []

    for target in target_feature_names:
        pred_col = f"prediction_{target}"

        if pred_col not in df.columns:
            continue

        for split_name in ["All", "Train", "Val", "Test"]:
            if split_name == "All":
                sub = df.copy()
            else:
                sub = df[df["Type"] == split_name].copy()

            valid = sub[[target, pred_col]].dropna()

            if len(valid) == 0:
                rows.append(
                    {
                        "target": target,
                        "split": split_name,
                        "n": 0,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MAPE": np.nan,
                        "R2": np.nan,
                        "PearsonR": np.nan,
                    }
                )
                continue

            y_true = valid[target].to_numpy(dtype=float)
            y_pred = valid[pred_col].to_numpy(dtype=float)

            error = y_pred - y_true

            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error ** 2))

            # --- MAPE ---
            nonzero_mask = y_true != 0
            if nonzero_mask.sum() > 0:
                mape = np.mean(
                    np.abs(error[nonzero_mask] / y_true[nonzero_mask])
                ) * 100
            else:
                mape = np.nan

            # --- R² (R-squared) ---
            ss_res = np.sum(error ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

            # --- Pearson 相关系数 ---
            if len(valid) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
                pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
            else:
                pearson_r = np.nan

            rows.append(
                {
                    "target": target,
                    "split": split_name,
                    "n": len(valid),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "R2": r2,
                    "PearsonR": pearson_r,
                }
            )

    return pd.DataFrame(rows)





def _fill_prediction_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    fillna_method: Optional[str],
) -> pd.DataFrame:
    """
    预测阶段使用和训练阶段一致的缺失值处理。
    """

    df = df.copy()

    if fillna_method == "zero":
        df[feature_cols] = df[feature_cols].fillna(0)

    elif fillna_method == "ffill":
        df[feature_cols] = df[feature_cols].ffill().bfill()

    elif fillna_method == "mean":
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    elif fillna_method == "drop":
        df = df.dropna(subset=feature_cols)

    elif fillna_method is None:
        pass

    else:
        raise ValueError(
            "fillna_method must be one of: 'zero', 'ffill', 'mean', 'drop', None."
        )

    return df



def train_torch_model(
    data: pd.DataFrame,
    Case_col: Union[str, List[str]],
    Feature_col: Union[str, List[str]],
    Time_col: str,
    model_config: Dict[str, Any],
):
    """
    Train a torch forecasting model and return predictions.

    Parameters
    ----------
    data:
        原始 DataFrame。

    Case_col:
        目标列名，可以是 str 或 list[str]。

    Feature_col:
        输入特征列名，可以是 str 或 list[str]。

    Time_col:
        时间列名。

    model_config:
        示例：

        {
            "modelType": "LSTM",
            "params": {
                "lookback": 12,
                "horizon": 3,
                "batch_size": 32,
                "dropout": 0.1,
            },
            "training_config": {
                "max_epochs": 100,
                "lr": 1e-3,
                "weight_decay": 1e-6,
                "patience": 10,
                "train_val_test_ratio": (6, 2, 2),
                "save_best_path": None,
            }
        }

    Returns
    -------
    output:
        dict with:
            model
            result_df
            metrics_df
            datamodule
    """

    # ============================================================
    # 0. Copy config，避免修改外部传入的 model_config
    # ============================================================
    model_config = copy.deepcopy(model_config)

    # ============================================================
    # 1. Normalize column inputs
    # ============================================================
    if isinstance(Case_col, str):
        target_feature_names = [Case_col]
    else:
        target_feature_names = list(Case_col)

    if isinstance(Feature_col, str):
        used_features = [Feature_col]
    else:
        used_features = list(Feature_col)

    if Time_col not in data.columns:
        raise ValueError(f"Time_col={Time_col!r} not found in data.columns.")

    missing_targets = [c for c in target_feature_names if c not in data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")

    missing_features = [c for c in used_features if c not in data.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in data: {missing_features}")

    # 去重，避免 target 和 feature 有重叠时重复选列
    selected_cols = list(dict.fromkeys([Time_col] + target_feature_names + used_features))

    data = data[selected_cols].copy().dropna()
    data[Time_col] = pd.to_datetime(data[Time_col])
    data = data.sort_values(Time_col).reset_index(drop=True)

    # ============================================================
    # 2. Resolve model config
    # ============================================================
    model_type = model_config.get("modelType", None)
    if model_type is None:
        raise ValueError("model_config['modelType'] is required.")

    modelname = f"{model_type}Forecaster"

    params = model_config.get("params", {})
    training_config_user = model_config.get("training_config", {})

    params = copy.deepcopy(params)
    training_config_user = copy.deepcopy(training_config_user)

    # ============================================================
    # 3. Defaults
    # ============================================================
    training_config_default = dict(
        max_epochs=100,
        lr=1e-3,
        weight_decay=1e-6,
        grad_clip_val=1.0,
        patience=10,
        min_delta=0.0,
        monitor="val_loss",
        monitor_mode="min",
        use_scheduler=False,
        scheduler_patience=5,
        scheduler_factor=0.5,
        save_best_path=None,
        print_every_epoch=True,
    )

    data_config_default = dict(
        train_val_test_ratio=(6, 2, 2),
        fillna_method="zero",
        normalize_x=True,
        normalize_y=True,
        ahead=0,
    )

    training_config = {**training_config_default, **training_config_user}

    train_val_test_ratio = training_config_user.get(
        "train_val_test_ratio",
        data_config_default["train_val_test_ratio"],
    )

    fillna_method = training_config_user.get(
        "fillna_method",
        data_config_default["fillna_method"],
    )

    normalize_x = training_config_user.get(
        "normalize_x",
        data_config_default["normalize_x"],
    )

    normalize_y = training_config_user.get(
        "normalize_y",
        data_config_default["normalize_y"],
    )

    ahead = training_config_user.get(
        "ahead",
        data_config_default["ahead"],
    )

    # 这些字段不属于 TrainConfig，不能传进去
    extra_data_keys = {
        "train_val_test_ratio",
        "fillna_method",
        "normalize_x",
        "normalize_y",
        "ahead",
    }

    train_config_kwargs = {
        k: v
        for k, v in training_config.items()
        if k not in extra_data_keys
    }

    # ============================================================
    # 4. Resolve model params
    # ============================================================
    batch_size = params.pop("batch_size", 32)
    lookback = params.pop("lookback", 12)
    horizon = params.pop("horizon", 3)

    input_dim = len(used_features)
    target_dim = len(target_feature_names)

    # ============================================================
    # 5. Build DataModule
    # ============================================================
    datamodule = SimpleForecastDataModule(
        df=data,
        target_feature_names=target_feature_names,
        input_feature_mode="explicit",
        input_feature_names=used_features,
        train_val_test_ratio=train_val_test_ratio,
        lookback=lookback,
        horizon=horizon,
        ahead=ahead,
        normalize_x=normalize_x,
        normalize_y=normalize_y,
        time_col=Time_col,
        city_col=None,
        city=None,
        fillna_method=fillna_method,
        batch_size=batch_size,
        train_shuffle=True,
    )

    datamodule.setup()

    # ============================================================
    # 6. Build model
    # ============================================================
    if modelname not in EpiAI_torch_models.__dict__:
        raise ValueError(
            f"Model {modelname!r} not found in EpiAI_torch_models."
        )

    model = EpiAI_torch_models.__dict__[modelname](
        lookback=lookback,
        horizon=horizon,
        input_dim=input_dim,
        target_dim=target_dim,
        **params,
    )

    # ============================================================
    # 7. Train
    # ============================================================
    loss_fn = training_config_user.get("loss_fn", torch.nn.MSELoss())

    config = TrainConfig(**train_config_kwargs)

    trained_model, history = fit_model(
        model=model,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
        loss_fn=loss_fn,
        config=config,
    )

    # ============================================================
    # 8. Prepare prediction input
    # ============================================================
    # 这里要保证预测时的缺失值处理和训练时一致
    pred_data = data[[Time_col] + used_features].copy()
    pred_data = _fill_prediction_features(
        df=pred_data,
        feature_cols=used_features,
        fillna_method=fillna_method,
    )

    pred_data = pred_data.set_index(Time_col)

    # ============================================================
    # 9. Predict
    # ============================================================
    result_df = predict_forecasts(
        model=trained_model,
        data=pred_data[used_features],
        input_feature_names=used_features,
        target_feature_names=target_feature_names,
        horizon=horizon,
        lookback=lookback,
        device=str(get_device()),
        target_normalizer=datamodule.bundle.y_scaler,
        input_normalizer=datamodule.bundle.x_scaler,
    )

    # ============================================================
    # 10. Merge prediction back to original data
    # ============================================================
    merged_df = data.copy()
    merged_df = merged_df.set_index(Time_col)

    merged_df = merged_df.join(result_df, how="left")

    # ============================================================
    # 11. Assign Train / Val / Test type
    # ============================================================
    merged_df["Type"] = None

    train_time_index = pd.to_datetime(datamodule.bundle.train_time_index)
    val_time_index = pd.to_datetime(datamodule.bundle.val_time_index)
    test_time_index = pd.to_datetime(datamodule.bundle.test_time_index)

    merged_df.loc[merged_df.index.isin(train_time_index), "Type"] = "Train"
    merged_df.loc[merged_df.index.isin(val_time_index), "Type"] = "Val"
    merged_df.loc[merged_df.index.isin(test_time_index), "Type"] = "Test"

    # ============================================================
    # 12. Evaluate metrics
    # ============================================================
    metrics_df = _evaluate_prediction_metrics(
        df=merged_df,
        target_feature_names=target_feature_names,
    )

    return {
        "model": trained_model,
        "result_df": merged_df.reset_index(drop=False),
        "metrics_df": metrics_df,
        "datamodule": datamodule,
    }




def train_tabular_model(
    data: pd.DataFrame,
    Case_col: Union[str, List[str]],
    Feature_col: Union[str, List[str]],
    Time_col: str,
    model_config: Dict[str, Any],
):
    """
    Train a non-torch / tabular forecasting model, such as XGB-style model.

    Parameters
    ----------
    data:
        原始 DataFrame。

    Case_col:
        目标列，可以是 str 或 list[str]。

    Feature_col:
        输入特征列，可以是 str 或 list[str]。

    Time_col:
        时间列。

    model_config:
        示例：

        {
            "modelType": "XGB",
            "params": {
                "lookback": 12,
                "horizon": 3,
                "batch_size": 32,
                ...
            },
            "training_config": {
                "train_val_test_ratio": (6, 2, 2),
                "fillna_method": "zero",
                "normalize_x": True,
                "normalize_y": True,
                "use_val": True,
            }
        }

    Returns
    -------
    output:
        dict with:
            model
            result_df
            metrics_df
            datamodule
    """

    # ============================================================
    # 0. Copy config，避免修改外部 model_config
    # ============================================================
    model_config = copy.deepcopy(model_config)

    # ============================================================
    # 1. Normalize column inputs
    # ============================================================
    if isinstance(Case_col, str):
        target_feature_names = [Case_col]
    else:
        target_feature_names = list(Case_col)

    if isinstance(Feature_col, str):
        used_features = [Feature_col]
    else:
        used_features = list(Feature_col)

    if Time_col not in data.columns:
        raise ValueError(f"Time_col={Time_col!r} not found in data.columns.")

    missing_targets = [c for c in target_feature_names if c not in data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")

    missing_features = [c for c in used_features if c not in data.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in data: {missing_features}")

    selected_cols = list(
        dict.fromkeys([Time_col] + target_feature_names + used_features)
    )

    data = data[selected_cols].copy()
    data[Time_col] = pd.to_datetime(data[Time_col])
    data = data.sort_values(Time_col).reset_index(drop=True)

    # ============================================================
    # 2. Resolve model config
    # ============================================================
    model_type = model_config.get("modelType", None)
    if model_type is None:
        raise ValueError("model_config['modelType'] is required.")

    modelname = f"{model_type}Forecaster"

    params = copy.deepcopy(model_config.get("params", {}))
    training_config_user = copy.deepcopy(model_config.get("training_config", {}))

    # ============================================================
    # 3. Defaults
    # ============================================================
    training_config_default = dict(
        train_val_test_ratio=(6, 2, 2),
        fillna_method="zero",
        normalize_x=True,
        normalize_y=True,
        ahead=0,
        use_val=True,
        fit_verbose=False,
    )

    training_config = {
        **training_config_default,
        **training_config_user,
    }

    train_val_test_ratio = training_config["train_val_test_ratio"]
    fillna_method = training_config["fillna_method"]
    normalize_x = training_config["normalize_x"]
    normalize_y = training_config["normalize_y"]
    ahead = training_config["ahead"]
    use_val = training_config["use_val"]
    fit_verbose = training_config["fit_verbose"]

    # ============================================================
    # 4. Resolve model params
    # ============================================================
    batch_size = params.pop("batch_size", 32)
    lookback = params.pop("lookback", 12)
    horizon = params.pop("horizon", 3)

    input_dim = len(used_features)
    target_dim = len(target_feature_names)

    # ============================================================
    # 5. Build DataModule
    # ============================================================
    datamodule = SimpleForecastDataModule(
        df=data,
        target_feature_names=target_feature_names,
        input_feature_mode="explicit",
        input_feature_names=used_features,

        train_val_test_ratio=train_val_test_ratio,

        lookback=lookback,
        horizon=horizon,
        ahead=ahead,

        normalize_x=normalize_x,
        normalize_y=normalize_y,

        time_col=Time_col,
        city_col=None,
        city=None,

        fillna_method=fillna_method,
        batch_size=batch_size,
        train_shuffle=False,
    )

    datamodule.setup()

    # ============================================================
    # 6. Build tabular model
    # ============================================================
    if modelname not in EpiAI_tabular_models.__dict__:
        raise ValueError(
            f"Model {modelname!r} not found in EpiAI_tabular_models."
        )

    model = EpiAI_tabular_models.__dict__[modelname](
        lookback=lookback,
        horizon=horizon,
        input_dim=input_dim,
        target_dim=target_dim,
        input_feature_names=datamodule.bundle.input_feature_names,
        **params,
    )

    # ============================================================
    # 7. Get all windows
    # ============================================================
    train_input, train_target = datamodule.bundle.train_dataset.get_all_windows()
    val_input, val_target = datamodule.bundle.val_dataset.get_all_windows()

    # train_input shape:
    #     (num_train_windows, lookback, input_dim)
    #
    # train_target shape:
    #     (num_train_windows, horizon, target_dim)

    # ============================================================
    # 8. Fit tabular model
    # ============================================================
    if use_val:
        model.fit(
            x=train_input,
            y=train_target,
            val_x=val_input,
            val_y=val_target,
            verbose=fit_verbose,
        )
    else:
        model.fit(
            x=train_input,
            y=train_target,
            verbose=fit_verbose,
        )

    # ============================================================
    # 9. Prepare prediction input
    # ============================================================
    pred_data = data[[Time_col] + used_features].copy()

    pred_data = _fill_prediction_features(
        df=pred_data,
        feature_cols=used_features,
        fillna_method=fillna_method,
    )

    pred_data = pred_data.set_index(Time_col)

    # ============================================================
    # 10. Predict
    # ============================================================
    result_df = predict_forecasts(
        model=model,
        data=pred_data[used_features],
        input_feature_names=used_features,
        target_feature_names=target_feature_names,

        horizon=horizon,
        lookback=lookback,

        device="cpu",

        target_normalizer=datamodule.bundle.y_scaler,
        input_normalizer=datamodule.bundle.x_scaler,

        # 如果你的 predict_forecasts 支持这个参数，
        # 且你的 XGB wrapper 接收 torch.Tensor：
        # non_torch_input_type="torch",
    )

    # ============================================================
    # 11. Merge prediction back to original data
    # ============================================================
    merged_df = data.copy()
    merged_df = merged_df.set_index(Time_col)

    merged_df = merged_df.join(result_df, how="left")

    # ============================================================
    # 12. Assign Train / Val / Test type
    # ============================================================
    merged_df["Type"] = None

    train_time_index = pd.to_datetime(datamodule.bundle.train_time_index)
    val_time_index = pd.to_datetime(datamodule.bundle.val_time_index)
    test_time_index = pd.to_datetime(datamodule.bundle.test_time_index)

    merged_df.loc[merged_df.index.isin(train_time_index), "Type"] = "Train"
    merged_df.loc[merged_df.index.isin(val_time_index), "Type"] = "Val"
    merged_df.loc[merged_df.index.isin(test_time_index), "Type"] = "Test"

    # ============================================================
    # 13. Evaluate metrics
    # ============================================================
    metrics_df = _evaluate_prediction_metrics(
        df=merged_df,
        target_feature_names=target_feature_names,
    )

    return {
        "model": model,
        "result_df": merged_df.reset_index(drop=False),
        "metrics_df": metrics_df,
        "datamodule": datamodule,
    }





def train_arima_model(
    data: pd.DataFrame,
    Case_col: Union[str, List[str]],
    Feature_col: Union[str, List[str]],
    Time_col: str,
    model_config: Dict[str, Any],
):
    """
    Train ARIMA/SARIMAX model with rolling-origin prediction.

    与 torch/tabular 不同：
    - 不切割滑窗，使用完整时间序列
    - train 上 auto_arima 搜索最优阶
    - test 上 rolling-origin 滚动预测
    - 支持 ARIMAX（外生变量）
    """

    # ============================================================
    # 1. Normalize column inputs
    # ============================================================
    model_config = copy.deepcopy(model_config)

    if isinstance(Case_col, str):
        target_feature_names = [Case_col]
    else:
        target_feature_names = list(Case_col)

    if isinstance(Feature_col, str):
        used_features = [Feature_col]
    else:
        used_features = list(Feature_col)

    if Time_col not in data.columns:
        raise ValueError(f"Time_col={Time_col!r} not found in data.columns.")

    missing_targets = [c for c in target_feature_names if c not in data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")

    missing_features = [c for c in used_features if c not in data.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in data: {missing_features}")

    # ARIMA 目前只支持单变量 target（target_dim=1）
    if len(target_feature_names) != 1:
        raise ValueError(
            f"ARIMA only supports target_dim=1, got target_dim={len(target_feature_names)}. "
            f"Target columns: {target_feature_names}"
        )

    target_col = target_feature_names[0]

    # ============================================================
    # 2. Resolve model config
    # ============================================================
    model_type = model_config.get("modelType", None)
    if model_type is None:
        raise ValueError("model_config['modelType'] is required.")
    modelname = f"{model_type}Forecaster"
    params = copy.deepcopy(model_config.get("params", {}))
    training_config_user = copy.deepcopy(model_config.get("training_config", {}))

    # ============================================================
    # 3. Defaults
    # ============================================================
    training_config_default = dict(
        train_val_test_ratio=(8, 0, 2),
        fillna_method="zero",
    )

    training_config = {**training_config_default, **training_config_user}

    train_val_test_ratio = training_config["train_val_test_ratio"]
    fillna_method = training_config["fillna_method"]

    # ============================================================
    # 4. Resolve params
    # ============================================================
    horizon = params.pop("horizon", 3)
    rolling_window_size = params.pop("rolling_window_size", None)

    # ============================================================
    # 5. Prepare data (完整序列，不切滑窗)
    # ============================================================
    selected_cols = list(
        dict.fromkeys([Time_col] + target_feature_names + used_features)
    )

    data = data[selected_cols].copy()
    data[Time_col] = pd.to_datetime(data[Time_col])
    data = data.sort_values(Time_col).reset_index(drop=True)

    # 缺失值填充
    if fillna_method == "zero":
        data = data.fillna(0)
    elif fillna_method == "ffill":
        data = data.ffill().bfill()
    elif fillna_method == "mean":
        data = data.fillna(data.mean())

    # 按时间顺序分成 train / test（val 只是占位符，不实际使用）
    total_len = len(data)
    train_ratio, _, test_ratio = train_val_test_ratio  # 只取第三个值
    total_ratio = train_ratio  + test_ratio
    test_len = int(total_len * test_ratio / total_ratio)


    train_end = total_len - test_len

    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]

    y_train = train_data[target_col].values.astype(float)
    y_test = test_data[target_col].values.astype(float)

    # ARIMAX: 如果 used_features 不包含 target 列，则作为外生变量
    X_features = [c for c in used_features if c != target_col]
    X_train = train_data[X_features].values.astype(float) if X_features else None
    X_test = test_data[X_features].values.astype(float) if X_features else None
    # ============================================================
    # 6. Build and fit model
    # ============================================================
    # 将 rolling_window_size 显式传给构造函数
    if rolling_window_size is not None:
        params["rolling_window_size"] = rolling_window_size
    
    if modelname not in EpiAI_arima_models.__dict__:         # ← 新增校验
        raise ValueError(
            f"Model {modelname!r} not found in EpiAI_arima_models. "
            f"Available models: {list(EpiAI_arima_models.__dict__.keys())}"
        )

    model = EpiAI_arima_models.__dict__[modelname](
        horizon=horizon,
        **params,
    )


    model.fit(y_train, X_train=X_train)
    # ============================================================
    # 7. Predict (rolling-origin on test)
    # ============================================================
    pred_df = model.predict(y_test, X_test=X_test, return_df=True)

    eval_data = test_data.copy()
    eval_data = eval_data.reset_index(drop=True)
    eval_data[f"prediction_{target_col}"] = pred_df["y_pred"].values

    # ============================================================
    # 8. 合并回完整数据
    # ============================================================
    merged_df = data.copy()

    # 将预测结果合并到原始时间索引
    time_map = dict(zip(eval_data[Time_col], eval_data[f"prediction_{target_col}"]))
    merged_df[f"prediction_{target_col}"] = merged_df[Time_col].map(time_map)

    # 标记 Train / Test（全部都是 Train，不标注 Val）
    merged_df["Type"] = None
    merged_df.loc[merged_df.index < train_end, "Type"] = "Train"
    merged_df.loc[merged_df.index >= train_end, "Type"] = "Test"


    # ============================================================
    # 9. Evaluate metrics
    # ============================================================
    metrics_df = _evaluate_prediction_metrics(
        df=merged_df,
        target_feature_names=target_feature_names,
    )

    return {
        "model": model,
        "result_df": merged_df.reset_index(drop=False),
        "metrics_df": metrics_df,
        "datamodule": None,  # ARIMA 不使用 DataModule
    }

