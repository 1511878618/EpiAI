"""
Sliding-window generation and city-dimension flattening for city-by-city training.
"""

from __future__ import annotations

from typing import Optional

import torch

from .containers import WindowedData


def make_sliding_windows(
    x: torch.Tensor,
    y: torch.Tensor,
    lookback: int,
    horizon: int,
    ahead: int = 0,
    mark: Optional[torch.Tensor] = None,
) -> WindowedData:
    """
    Generate sliding-window forecasting samples.

    Input shapes
    ------------
    x:
        (time, city, input_dim)
    y:
        (time, city, target_dim)
    mark:
        (time, city, mark_dim) or None

    Output shapes
    -------------
    window_x:
        (num_samples, lookback, city, input_dim)
    window_y:
        (num_samples, horizon, city, target_dim)
    window_x_mark:
        (num_samples, lookback, city, mark_dim) or None
    window_y_mark:
        (num_samples, horizon, city, mark_dim) or None
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

    if mark is not None:
        if mark.dim() != 3:
            raise ValueError(f"`mark` must be 3D with shape (time, city, mark_dim), got {tuple(mark.shape)}.")
        if mark.shape[0] != x.shape[0] or mark.shape[1] != x.shape[1]:
            raise ValueError(
                f"mark must align with x on time and city dimensions, "
                f"got mark={tuple(mark.shape)}, x={tuple(x.shape)}."
            )

    total_window = lookback + ahead + horizon
    if x.shape[0] < total_window:
        raise ValueError(
            f"Time length {x.shape[0]} is shorter than required total window {total_window}."
        )

    indices = [(i, i + total_window) for i in range(x.shape[0] - total_window + 1)]

    x_windows = []
    y_windows = []
    x_mark_windows = [] if mark is not None else None
    y_mark_windows = [] if mark is not None else None

    for i, j in indices:
        hist_start = i
        hist_end = i + lookback
        fut_start = i + lookback + ahead
        fut_end = j

        # x: [lookback, city, input_dim]
        x_windows.append(x[hist_start:hist_end])

        # y: [horizon, city, target_dim]
        y_windows.append(y[fut_start:fut_end])

        if mark is not None:
            # x_mark: [lookback, city, mark_dim]
            x_mark_hist = mark[hist_start:hist_end]
            x_mark_windows.append(x_mark_hist)

            # y_mark: future horizon markers
            fut_mark = mark[fut_start:fut_end]
            y_mark = fut_mark
            y_mark_windows.append(y_mark)

    window_x = torch.stack(x_windows) if x_windows else torch.empty(0)
    window_y = torch.stack(y_windows) if y_windows else torch.empty(0)

    if x_mark_windows is not None:
        window_x_mark = torch.stack(x_mark_windows) if x_mark_windows else torch.empty(0)
        window_y_mark = torch.stack(y_mark_windows) if y_mark_windows else torch.empty(0)
    else:
        window_x_mark = None
        window_y_mark = None

    return WindowedData(
        x=window_x,
        y=window_y,
        x_mark=window_x_mark,
        y_mark=window_y_mark,
    )

def flatten_city_windows_for_training(
    window_x: torch.Tensor,
    window_y: torch.Tensor,
    window_x_mark: Optional[torch.Tensor] = None,
    window_y_mark: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Flatten city dimension into sample dimension for city-by-city training.

    Input shapes
    ------------
    window_x:
        (num_samples, lookback, city, input_dim)
    window_y:
        (num_samples, horizon, city, target_dim)
    window_x_mark:
        (num_samples, lookback, city, mark_dim) or None
    window_y_mark:
        (num_samples, horizon, city, mark_dim) or None

    Output shapes
    -------------
    model_input:
        (city * num_samples, lookback, input_dim)
    model_target:
        (city * num_samples, horizon, target_dim)
    model_x_mark:
        (city * num_samples, lookback, mark_dim) or None
    model_y_mark:
        (city * num_samples, horizon, mark_dim) or None
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

    if window_x_mark is not None:
        if window_x_mark.dim() != 4:
            raise ValueError(f"`window_x_mark` must be 4D, got {tuple(window_x_mark.shape)}.")
        m_num_samples, m_lookback, m_city, mark_dim = window_x_mark.shape
        if m_num_samples != num_samples or m_lookback != lookback or m_city != city:
            raise ValueError(
                f"Incompatible shapes: window_x={tuple(window_x.shape)}, window_x_mark={tuple(window_x_mark.shape)}."
            )

    if window_y_mark is not None:
        if window_y_mark.dim() != 4:
            raise ValueError(f"`window_y_mark` must be 4D, got {tuple(window_y_mark.shape)}.")
        ym_num_samples, y_mark_len, ym_city, y_mark_dim = window_y_mark.shape
        if ym_num_samples != num_samples or ym_city != city:
            raise ValueError(
                f"Incompatible shapes: window_x={tuple(window_x.shape)}, window_y_mark={tuple(window_y_mark.shape)}."
            )

    model_input = window_x.permute(2, 0, 1, 3).reshape(city * num_samples, lookback, input_dim)
    model_target = window_y.permute(2, 0, 1, 3).reshape(city * num_samples, horizon, target_dim)

    if window_x_mark is not None:
        model_x_mark = window_x_mark.permute(2, 0, 1, 3).reshape(city * num_samples, lookback, mark_dim)
    else:
        model_x_mark = None

    if window_y_mark is not None:
        model_y_mark = window_y_mark.permute(2, 0, 1, 3).reshape(city * num_samples, y_mark_len, y_mark_dim)
    else:
        model_y_mark = None

    return model_input, model_target, model_x_mark, model_y_mark


__all__ = ["make_sliding_windows", "flatten_city_windows_for_training"]
