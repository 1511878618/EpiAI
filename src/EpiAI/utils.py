from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext

def predict_forecasts(
    model,
    data,
    horizon: int,
    lookback: int,
    device: str = "cpu",
    input_feature_names: Optional[list[str]] = None,
    target_feature_names: Optional[list[str]] = None,
    input_normalizer=None,
    target_normalizer=None,
    target_col: Optional[str] = None,
    non_torch_input_type: Literal["numpy", "torch"] = "numpy",
):
    """Rolling multi-step forecasting for time-series models.

    对完整的时间序列数据执行滑窗预测，每个窗口向前预测 horizon 步，
    窗口步长 = horizon（无重叠）。

    Parameters
    ----------
    model : torch.nn.Module | object
        已训练好的预测模型。支持：
        - PyTorch 模型（自动切换到 eval 模式、移动到 device）
        - 实现了 .predict() 或 .__call__() 的非 torch 模型
    data : pd.DataFrame | pd.Series | np.ndarray | torch.Tensor | list
        输入数据。不同格式的处理方式：

        - pd.DataFrame:
            * 如果提供 input_feature_names，用对应列作为输入
            * 如果提供 target_col，用该列作为输入
            * 否则自动选择所有数值列
        - pd.Series / np.ndarray / torch.Tensor / list:
            直接转换为 (time, input_dim) 的 float32 张量

    horizon : int
        每个窗口的预测步数（> 0）。
    lookback : int
        每个窗口的历史步数（> 0）。
    device : str, default="cpu"
        推理设备，仅对 PyTorch 模型有效。
    input_feature_names : list[str], optional
        当 data 是 DataFrame 时，指定用作输入的列名。
    target_feature_names : list[str], optional
        输出列名，用于构建结果 DataFrame 的列标题。
        默认：target_dim=1 时用 ["target"]，否则用 ["target_0", ...]。
    input_normalizer : object, optional
        输入归一化器，须实现 .transform() 方法。
        在预测前对输入数据做反归一化。
    target_normalizer : object, optional
        输出反归一化器，须实现 .inverse_transform() 方法。
        在预测后对输出做反归一化。
    target_col : str, optional
        当 data 是 DataFrame 且未提供 input_feature_names 时，
        指定目标列名（自动用该列作为输入）。
    non_torch_input_type : Literal["numpy", "torch"], default="numpy"
        非 torch 模型期望的输入类型。
        - "numpy": 输入转为 np.ndarray
        - "torch": 输入保持 torch.Tensor

    Returns
    -------
    result_df : pd.DataFrame
        shape = (num_windows * horizon, target_dim)
        列说明：
            index          时间点索引（原始 index 或数值位置）
            prediction_*   各目标变量的预测值（反归一化后）

    Raises
    ------
    ValueError
        - horizon 或 lookback 未提供 / 非法
        - 输入数据所有列非数值且未指定 input_feature_names
        - 模型输出形状无法解析
        - 没有生成任何有效窗口

    滑窗逻辑
    --------
    每个窗口的输入形状为 (1, lookback, input_dim)：

        for i in range(0, len(data) - horizon - lookback, horizon):
            history = data[i : i + lookback]       # (lookback, input_dim)
            model_input = history.unsqueeze(0)      # (1, lookback, input_dim)
            output = model(model_input)             # (1, horizon, target_dim)

    Notes
    -----
    - 窗口步长固定为 horizon，不会出现重叠窗口。
    - 仅输出预测值，不返回 ground truth。
    - 支持 torch.no_grad() 自动上下文管理。
    """


    if horizon is None or lookback is None:
        raise ValueError("`horizon` and `lookback` must be provided.")

    if horizon <= 0:
        raise ValueError(f"`horizon` must be > 0, got {horizon}.")

    if lookback <= 0:
        raise ValueError(f"`lookback` must be > 0, got {lookback}.")

    # ============================================================
    # 1. 判断是不是 torch 模型
    # ============================================================
    is_torch_model = isinstance(model, torch.nn.Module)

    if is_torch_model:
        model = model.to(device)
        model.eval()

    # ============================================================
    # 2. 处理 data -> numpy array: (time, input_dim)
    # ============================================================
    if isinstance(data, pd.Series):
        data_values = data.values
        original_index = data.index

    elif isinstance(data, pd.DataFrame):
        original_index = data.index

        if input_feature_names is not None:
            missing_cols = [
                col for col in input_feature_names
                if col not in data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"input_feature_names not found in data.columns: {missing_cols}"
                )

            data_values = data[input_feature_names].values

        elif target_col is not None:
            if target_col not in data.columns:
                raise ValueError(f"target_col={target_col!r} not found in data.columns.")

            data_values = data[target_col].values

        else:
            data_values = data.select_dtypes(include=[np.number]).values

            if data_values.shape[1] == 0:
                raise ValueError(
                    "No numeric columns found in DataFrame. "
                    "Please provide `input_feature_names`."
                )

    elif isinstance(data, np.ndarray):
        data_values = data
        original_index = None

    elif isinstance(data, torch.Tensor):
        data_values = data.detach().cpu().numpy()
        original_index = None

    else:
        data_values = np.array(data)
        original_index = None

    data_values = np.asarray(data_values, dtype=np.float32)

    if data_values.ndim == 1:
        data_values = data_values[:, None]

    if data_values.ndim != 2:
        raise ValueError(
            f"data must be 2D after processing: (time, input_dim), "
            f"got shape={data_values.shape}."
        )

    # ============================================================
    # 3. 输入归一化
    # ============================================================
    if input_normalizer is not None:
        data_values = input_normalizer.transform(data_values)

        if isinstance(data_values, torch.Tensor):
            data_values = data_values.detach().cpu().numpy()

        data_values = np.asarray(data_values, dtype=np.float32)

    data_tensor = torch.as_tensor(data_values, dtype=torch.float32)

    # ============================================================
    # 4. 输出整理 helper
    # ============================================================
    def to_numpy_output(output) -> np.ndarray:
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().numpy()

        return np.asarray(output)

    def format_prediction_output(output) -> np.ndarray:
        """
        把模型输出统一整理成：

            (horizon, target_dim)
        """
        out = to_numpy_output(output)
        out = np.asarray(out, dtype=np.float32)

        # 常见 torch 输出: (1, horizon, target_dim)
        if out.ndim == 3:
            if out.shape[0] != 1:
                raise ValueError(
                    f"Only batch size 1 is supported, got output shape={out.shape}."
                )
            out = out[0]

        # 常见 sklearn / XGB wrapper 输出: (1, horizon) 或 (1, horizon, target_dim)
        elif out.ndim == 2:
            if out.shape[0] == 1 and out.shape[1] == horizon:
                out = out.reshape(horizon, 1)

            elif out.shape[0] == horizon:
                # 已经是 (horizon, target_dim)
                pass

            else:
                flat = out.reshape(-1)

                if target_feature_names is None:
                    if flat.size == horizon:
                        out = flat.reshape(horizon, 1)
                    else:
                        raise ValueError(
                            "Cannot infer target_dim from model output. "
                            "Please provide target_feature_names."
                        )
                else:
                    target_dim = len(target_feature_names)
                    expected_size = horizon * target_dim

                    if flat.size != expected_size:
                        raise ValueError(
                            f"Cannot reshape output with shape {out.shape} to "
                            f"(horizon={horizon}, target_dim={target_dim})."
                        )

                    out = flat.reshape(horizon, target_dim)

        elif out.ndim == 1:
            if target_feature_names is None:
                if out.size == horizon:
                    out = out.reshape(horizon, 1)
                else:
                    raise ValueError(
                        "Cannot infer target_dim from 1D model output. "
                        "Please provide target_feature_names."
                    )
            else:
                target_dim = len(target_feature_names)
                expected_size = horizon * target_dim

                if out.size != expected_size:
                    raise ValueError(
                        f"Cannot reshape output of size {out.size} to "
                        f"(horizon={horizon}, target_dim={target_dim})."
                    )

                out = out.reshape(horizon, target_dim)

        else:
            raise ValueError(f"Unsupported model output shape: {out.shape}.")

        if out.shape[0] != horizon:
            raise ValueError(
                f"Prediction horizon mismatch. "
                f"Expected horizon={horizon}, got output.shape={out.shape}."
            )

        return out

    def inverse_transform_prediction(pred_np: np.ndarray) -> np.ndarray:
        """
        pred_np shape: (horizon, target_dim)
        """
        if target_normalizer is None:
            return pred_np

        original_shape = pred_np.shape
        pred_2d = pred_np.reshape(-1, original_shape[-1])

        pred_inv = target_normalizer.inverse_transform(pred_2d)

        if isinstance(pred_inv, torch.Tensor):
            pred_inv = pred_inv.detach().cpu().numpy()

        pred_inv = np.asarray(pred_inv, dtype=np.float32)

        return pred_inv.reshape(original_shape)

    # ============================================================
    # 5. 滑窗预测
    # ============================================================
    prediction_chunks = []
    label_indices = []

    context = torch.no_grad() if is_torch_model else nullcontext()

    with context:
        for i in range(0, len(data_tensor) - horizon - lookback, horizon):
            history = data_tensor[i:i + lookback]

            # 所有模型统一使用这个逻辑：
            # shape: (1, lookback, input_dim)
            history_batch = history.unsqueeze(0)

            if is_torch_model:
                model_input = history_batch.to(device)

                if hasattr(model, "predict"):
                    output = model.predict(model_input)
                else:
                    output = model(model_input)

            else:
                if non_torch_input_type == "numpy":
                    model_input = history_batch.detach().cpu().numpy()
                elif non_torch_input_type == "torch":
                    model_input = history_batch
                else:
                    raise ValueError(
                        "non_torch_input_type must be either 'numpy' or 'torch'."
                    )

                if hasattr(model, "predict"):
                    output = model.predict(model_input)
                else:
                    output = model(model_input)

            pred_np = format_prediction_output(output)
            pred_np = inverse_transform_prediction(pred_np)

            preds = torch.as_tensor(pred_np, dtype=torch.float32)
            prediction_chunks.append(preds)

            # 和你原始逻辑一致：
            # 预测对应 i + lookback 到 i + lookback + horizon - 1
            label_indices.extend(range(i + lookback, i + lookback + horizon))

    if not prediction_chunks:
        raise ValueError("No valid forecasting windows were generated.")

    # ============================================================
    # 6. 整理结果
    # ============================================================
    predictions = torch.cat(prediction_chunks, dim=0)

    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(-1)

    target_dim = predictions.shape[1]

    if target_feature_names is None:
        if target_dim == 1:
            target_feature_names = ["target"]
        else:
            target_feature_names = [f"target_{i}" for i in range(target_dim)]

    if len(target_feature_names) != target_dim:
        raise ValueError(
            f"len(target_feature_names) must match model output target_dim. "
            f"Got len(target_feature_names)={len(target_feature_names)}, "
            f"target_dim={target_dim}."
        )

    if original_index is not None and len(original_index) > max(label_indices):
        result_index = [original_index[i] for i in label_indices]
    else:
        result_index = label_indices

    result_dict = {
        "index": result_index,
    }

    pred_np = predictions.numpy()

    for j, name in enumerate(target_feature_names):
        result_dict[f"prediction_{name}"] = pred_np[:, j]

    result_df = pd.DataFrame(result_dict).set_index("index")

    return result_df


