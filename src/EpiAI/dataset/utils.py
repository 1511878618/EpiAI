"""
Generic utilities that do not fit anywhere else.
"""
from __future__ import annotations


from typing import Optional, Sequence
import pandas as pd

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





def build_tensor_data_from_dataframe(
    df: pd.DataFrame,
    time_col: str = "Year/Month",
    city_col: str = "province",
    feature_cols: Optional[Sequence[str]] = None,
    sort_index: bool = True,
    dtype: torch.dtype = torch.float32,
    check_duplicates: bool = True,
    check_missing: bool = True,
) -> dict:
    """
    Convert a long-format DataFrame into a tensor-based dictionary with shape
    (time, city, feature), together with dimension names and coordinate metadata.

    Parameters
    ----------
    df:
        Input DataFrame in long format. Each row should correspond to one
        (time, city) pair, with multiple feature columns.
    time_col:
        Name of the time column.
    city_col:
        Name of the city/province column.
    feature_cols:
        Feature columns to include. If None, all columns except `time_col`
        and `city_col` are used.
    sort_index:
        Whether to sort the multi-index before conversion.
    dtype:
        Torch dtype of the output tensor.
    check_duplicates:
        Whether to check duplicated (time, city) rows.
    check_missing:
        Whether to check missing values in the selected columns.

    Returns
    -------
    save_dict:
        A dictionary with the following keys:
        - "tensor": torch.Tensor of shape (time, city, feature)
        - "dims": list of dimension names
        - "coords": dict containing coordinate lists for each dimension

    Notes
    -----
    The output format is designed to match the dataset layer input:
        tensor.shape = (time, city, feature)

    Example
    -------
    data
    

    save_dict = build_tensor_data_from_dataframe(
        df=data,
        time_col="Year/Month",
        city_col="province",
    )
    """
    # -------------------------------------------------------------------------
    # Validate required columns.
    # -------------------------------------------------------------------------
    required_cols = {time_col, city_col}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    # -------------------------------------------------------------------------
    # Determine feature columns.
    # -------------------------------------------------------------------------
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in {time_col, city_col}]
    else:
        feature_cols = list(feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("No feature columns were selected.")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")

    selected_cols = [time_col, city_col] + feature_cols
    df_selected = df[selected_cols].copy()

    # -------------------------------------------------------------------------
    # Check duplicated (time, city) keys.
    # -------------------------------------------------------------------------
    if check_duplicates:
        duplicated_mask = df_selected.duplicated(subset=[time_col, city_col], keep=False)
        if duplicated_mask.any():
            duplicated_rows = df_selected.loc[duplicated_mask, [time_col, city_col]]
            raise ValueError(
                "Duplicated (time, city) rows detected. "
                "Each (time, city) pair must correspond to exactly one row.\n"
                f"Example duplicates:\n{duplicated_rows.head()}"
            )

    # -------------------------------------------------------------------------
    # Check missing values in selected columns.
    # -------------------------------------------------------------------------
    if check_missing:
        if df_selected[feature_cols].isnull().any().any():
            missing_summary = df_selected[feature_cols].isnull().sum()
            missing_summary = missing_summary[missing_summary > 0]
            raise ValueError(
                "Missing values detected in selected feature columns:\n"
                f"{missing_summary}"
            )

    # -------------------------------------------------------------------------
    # Convert to xarray with dimensions (time, city, feature).
    # -------------------------------------------------------------------------
    xr_data = df_selected.set_index([time_col, city_col])
    if sort_index:
        xr_data = xr_data.sort_index()

    xr_data = xr_data.to_xarray()

    xr_data_array = xr_data.to_array().transpose(time_col, city_col, "variable")

    # -------------------------------------------------------------------------
    # Build the output dictionary.
    # -------------------------------------------------------------------------
    save_dict = {
        "tensor": torch.tensor(xr_data_array.values, dtype=dtype),
        "dims": list(xr_data_array.dims),
        "coords": {
            time_col: xr_data_array.coords[time_col].values.tolist(),
            city_col: xr_data_array.coords[city_col].values.tolist(),
            "feature": xr_data_array.coords["variable"].values.tolist(),
        },
    }

    return save_dict
