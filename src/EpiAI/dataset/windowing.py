"""
Sliding-window generation and city-dimension flattening for city-by-city training.
"""
from __future__ import annotations

import torch

from .containers import WindowedData


def make_sliding_windows(
    x: torch.Tensor,
    y: torch.Tensor,
    lookback: int,
    horizon: int,
    ahead: int = 0,
) -> WindowedData:
    """
    Generate sliding-window forecasting samples.

    Input shapes
    ------------
    x : (time, city, input_dim)
    y : (time, city, target_dim)

    Output shapes
    -------------
    window_x : (num_samples, lookback, city, input_dim)
    window_y : (num_samples, horizon,  city, target_dim)
    """
    if x.dim() != 3:
        raise ValueError(f"`x` must be 3D with shape (time, city, input_dim), got {tuple(x.shape)}.")
    if y.dim() != 3:
        raise ValueError(f"`y` must be 3D with shape (time, city, target_dim), got {tuple(y.shape)}.")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must align on time dimension, got x.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}."
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must align on city dimension, got x.shape[1]={x.shape[1]}, y.shape[1]={y.shape[1]}."
        )

    total_window = lookback + ahead + horizon
    if x.shape[0] < total_window:
        raise ValueError(
            f"Time length {x.shape[0]} is shorter than required total window {total_window}."
        )

    indices = [(i, i + total_window) for i in range(x.shape[0] - total_window + 1)]

    x_windows = []
    y_windows = []

    for i, j in indices:
        x_windows.append(x[i:i + lookback])
        y_windows.append(y[i + lookback + ahead:j])

    window_x = torch.stack(x_windows) if x_windows else torch.empty(0)
    window_y = torch.stack(y_windows) if y_windows else torch.empty(0)

    return WindowedData(x=window_x, y=window_y)


def flatten_city_windows_for_training(
    window_x: torch.Tensor,
    window_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten city dimension into sample dimension for city-by-city training.

    Input shapes
    ------------
    window_x : (num_samples, lookback, city, input_dim)
    window_y : (num_samples, horizon,  city, target_dim)

    Output shapes
    -------------
    model_input  : (city * num_samples, lookback, input_dim)
    model_target : (city * num_samples, horizon,  target_dim)
    """
    if window_x.dim() != 4:
        raise ValueError(f"`window_x` must be 4D, got {tuple(window_x.shape)}.")
    if window_y.dim() != 4:
        raise ValueError(f"`window_y` must be 4D, got {tuple(window_y.shape)}.")

    num_samples, lookback, city, input_dim = window_x.shape
    y_num_samples, horizon, y_city, target_dim = window_y.shape

    if num_samples != y_num_samples or city != y_city:
        raise ValueError(
            f"Incompatible shapes: window_x={tuple(window_x.shape)}, window_y={tuple(window_y.shape)}."
        )

    model_input = window_x.permute(2, 0, 1, 3).reshape(city * num_samples, lookback, input_dim)
    model_target = window_y.permute(2, 0, 1, 3).reshape(city * num_samples, horizon, target_dim)

    return model_input, model_target


__all__ = ["make_sliding_windows", "flatten_city_windows_for_training"]
