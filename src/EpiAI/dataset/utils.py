"""
Generic utilities that do not fit anywhere else.
"""
from __future__ import annotations

from typing import Optional

import torch

def shuffle_training_data(
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    train_x_mark: Optional[torch.Tensor] = None,
    train_y_mark: Optional[torch.Tensor] = None,
    seed: int = 42,
):
    """
    Shuffle training samples along the first dimension while keeping
    input-target-(mark) alignment.
    """
    num_samples = train_input.shape[0]

    if train_target.shape[0] != num_samples:
        raise ValueError("train_input and train_target must have same number of samples")

    if train_x_mark is not None and train_x_mark.shape[0] != num_samples:
        raise ValueError("train_x_mark must have same number of samples as train_input")

    if train_y_mark is not None and train_y_mark.shape[0] != num_samples:
        raise ValueError("train_y_mark must have same number of samples as train_input")

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)

    train_input = train_input[perm]
    train_target = train_target[perm]

    if train_x_mark is not None:
        train_x_mark = train_x_mark[perm]

    if train_y_mark is not None:
        train_y_mark = train_y_mark[perm]

    return train_input, train_target, train_x_mark, train_y_mark