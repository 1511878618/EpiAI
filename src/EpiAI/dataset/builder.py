"""
End-to-end dataset builder that wires every stage together.
"""
from __future__ import annotations

from .config import DatasetConfig
from .containers import DatasetBundle
from .io import load_disease_tensor
from .normalizer import normalize_split_data
from .splitter import CitySplitter
from .task_builder import FeatureTaskBuilder
from .utils import shuffle_training_data
from .windowing import flatten_city_windows_for_training, make_sliding_windows

class MultiTargetCityDatasetBuilder:
    """
    End-to-end dataset builder for multi-target city-by-city forecasting.

    Pipeline
    --------
    1. load raw tensor
    2. build x/y(/mark) from feature dimension
    3. split by city
    4. fit normalizers on train split only
    5. generate sliding windows
    6. flatten city dimension into sample dimension
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def build(self) -> DatasetBundle:
        raw_data = load_disease_tensor(self.config.data_path)

        task_builder = FeatureTaskBuilder(raw_data)
        raw_x, raw_y, raw_mark, metadata = task_builder.build_xy(
            target_feature_names=self.config.target_feature_names,
            input_feature_mode=self.config.input_feature_mode,
            input_feature_names=self.config.input_feature_names,
            mark_feature_names=self.config.mark_feature_names,
            remove_mark_from_input=self.config.remove_mark_from_input,
        )

        splitter = CitySplitter(city_dim=self.config.city_dim)
        split_data = splitter.split(
            x=raw_x,
            y=raw_y,
            mark=raw_mark,
            split_mode=self.config.split_mode,
            train_val_test_cutoff_line=self.config.train_val_test_cutoff_line,
            train_city_indices=self.config.train_city_indices,
            val_city_indices=self.config.val_city_indices,
            test_city_indices=self.config.test_city_indices,
        )

        split_data, x_normalizer, y_normalizer = normalize_split_data(
            split_data=split_data,
            normalize_x=self.config.normalize_x,
            normalize_y=self.config.normalize_y,
            x_norm_dims=self.config.x_norm_dims,
            y_norm_dims=self.config.y_norm_dims,
        )

        label_len = self.config.resolve_label_len

        train_windows = make_sliding_windows(
            x=split_data.x_train,
            y=split_data.y_train,
            lookback=self.config.lookback,
            horizon=self.config.horizon,
            ahead=self.config.ahead,
            mark=split_data.mark_train,
            # label_len=label_len,   # NEW
        )
        val_windows = make_sliding_windows(
            x=split_data.x_val,
            y=split_data.y_val,
            lookback=self.config.lookback,
            horizon=self.config.horizon,
            ahead=self.config.ahead,
            mark=split_data.mark_val,
            # label_len=label_len,   # NEW
        )
        test_windows = make_sliding_windows(
            x=split_data.x_test,
            y=split_data.y_test,
            lookback=self.config.lookback,
            horizon=self.config.horizon,
            ahead=self.config.ahead,
            mark=split_data.mark_test,
            # label_len=label_len,   # NEW
        )

        train_input, train_target, train_x_mark, train_y_mark = flatten_city_windows_for_training(
            train_windows.x,
            train_windows.y,
            train_windows.x_mark,
            train_windows.y_mark,
        )
        val_input, val_target, val_x_mark, val_y_mark = flatten_city_windows_for_training(
            val_windows.x,
            val_windows.y,
            val_windows.x_mark,
            val_windows.y_mark,
        )
        test_input, test_target, test_x_mark, test_y_mark = flatten_city_windows_for_training(
            test_windows.x,
            test_windows.y,
            test_windows.x_mark,
            test_windows.y_mark,
        )

        if self.config.shuffle_train:
            train_input, train_target, train_x_mark, train_y_mark = shuffle_training_data(
                train_input=train_input,
                train_target=train_target,
                train_x_mark=train_x_mark,
                train_y_mark=train_y_mark,   # NEW
                seed=self.config.shuffle_seed,
            )

        all_city_names = list(metadata["province_names"])
        city_name_dict = {
            "train": [all_city_names[i] for i in split_data.train_city_indices],
            "val": [all_city_names[i] for i in split_data.val_city_indices],
            "test": [all_city_names[i] for i in split_data.test_city_indices],
        }

        return DatasetBundle(
            train_input=train_input,
            train_target=train_target,
            train_x_mark=train_x_mark,
            train_y_mark=train_y_mark,   # NEW

            val_input=val_input,
            val_target=val_target,
            val_x_mark=val_x_mark,
            val_y_mark=val_y_mark,       # NEW

            test_input=test_input,
            test_target=test_target,
            test_x_mark=test_x_mark,
            test_y_mark=test_y_mark,     # NEW

            raw_x=raw_x,
            raw_y=raw_y,
            raw_mark=raw_mark,
            split_data=split_data,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            city_name_dict=city_name_dict,
            all_city_names=all_city_names,
            time_index=list(metadata["time_index"]),
            input_feature_names=list(metadata["input_feature_names"]),
            target_feature_names=list(metadata["target_feature_names"]),
            input_feature_indices=list(metadata["input_feature_indices"]),
            target_feature_indices=list(metadata["target_feature_indices"]),
            mark_feature_names=list(metadata["mark_feature_names"]),
            mark_feature_indices=list(metadata["mark_feature_indices"]),
        )


__all__ = ["MultiTargetCityDatasetBuilder"]
