"""
Generic utilities that do not fit anywhere else.
"""
from __future__ import annotations

from typing import Optional

import torch


def shuffle_training_data(
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shuffle training samples along the first dimension while keeping
    input-target alignment.

    Parameters
    ----------
    train_input:
        Tensor of shape (N, lookback, input_dim)
    train_target:
        Tensor of shape (N, horizon, target_dim)
    seed:
        Optional random seed for reproducible shuffling.

    Returns
    -------
    shuffled_input, shuffled_target
    """
    if train_input.shape[0] != train_target.shape[0]:
        raise ValueError(
            f"train_input and train_target must have the same number of samples, "
            f"got {train_input.shape[0]} and {train_target.shape[0]}."
        )

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(train_input.shape[0], generator=generator)
    else:
        perm = torch.randperm(train_input.shape[0])

    return train_input[perm], train_target[perm]


__all__ = ["shuffle_training_data"]
