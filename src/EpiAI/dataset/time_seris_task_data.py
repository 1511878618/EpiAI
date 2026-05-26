from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


InputFeatureMode = Literal["all", "exclude_targets", "explicit"]


# ============================================================
# 1. 简单 StandardScaler
# ============================================================

class SimpleStandardScaler:
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "SimpleStandardScaler":
        self.mean_ = np.nanmean(x, axis=0, keepdims=True)
        self.std_ = np.nanstd(x, axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return x * self.std_ + self.mean_


# ============================================================
# 2. Dataset：把连续时间序列切成窗口
# ============================================================

class TimeSeriesWindowDataset(Dataset):
    """
    X_seq: (time, input_dim)
    y_seq: (time, target_dim)

    返回：
        x_window: (lookback, input_dim)
        y_window: (horizon, target_dim)
    """

    def __init__(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        lookback: int,
        horizon: int,
        ahead: int = 0,
    ) -> None:
        super().__init__()

        if X_seq.ndim != 2:
            raise ValueError(f"X_seq must be 2D: (time, input_dim), got {X_seq.shape}")

        if y_seq.ndim != 2:
            raise ValueError(f"y_seq must be 2D: (time, target_dim), got {y_seq.shape}")

        if len(X_seq) != len(y_seq):
            raise ValueError(
                f"X_seq and y_seq must have same time length, "
                f"got {len(X_seq)} and {len(y_seq)}."
            )

        if lookback <= 0:
            raise ValueError("lookback must be > 0")

        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        if ahead < 0:
            raise ValueError("ahead must be >= 0")

        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)

        self.lookback = lookback
        self.horizon = horizon
        self.ahead = ahead

        self.max_start = len(X_seq) - lookback - ahead - horizon + 1

        if self.max_start <= 0:
            raise ValueError(
                "Sequence is too short for given lookback, ahead and horizon. "
                f"Got time length={len(X_seq)}, lookback={lookback}, "
                f"ahead={ahead}, horizon={horizon}."
            )

    def __len__(self) -> int:
        return self.max_start

    def __getitem__(self, idx: int):
        x_start = idx
        x_end = idx + self.lookback

        y_start = x_end + self.ahead
        y_end = y_start + self.horizon

        x = self.X_seq[x_start:x_end]
        y = self.y_seq[y_start:y_end]

        return x, y

    def get_all_windows(self):
        """
        一次性返回所有滑窗样本。

        Returns
        -------
        X:
            shape = (num_windows, lookback, input_dim)

        y:
            shape = (num_windows, horizon, target_dim)
        """
        xs = []
        ys = []

        for idx in range(len(self)):
            x, y = self[idx]
            xs.append(x)
            ys.append(y)

        X = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)

        return X, y

# ============================================================
# 3. 保存构建结果，方便之后查看
# ============================================================

@dataclass
class ForecastDataBundle:
    train_dataset: TimeSeriesWindowDataset
    val_dataset: TimeSeriesWindowDataset
    test_dataset: TimeSeriesWindowDataset

    input_feature_names: list[str]
    target_feature_names: list[str]

    train_time_index: list
    val_time_index: list
    test_time_index: list

    x_scaler: Optional[SimpleStandardScaler]
    y_scaler: Optional[SimpleStandardScaler]


# ============================================================
# 4. Lightning DataModule
# ============================================================

class SimpleForecastDataModule(L.LightningDataModule):
    """
    简单时间序列预测 DataModule。

    输入 DataFrame 格式类似：

        Year/Month | City | Dengue fever | Influenza | ...

    最终 DataLoader 每个 batch 返回：

        batch_x: (batch_size, lookback, input_dim)
        batch_y: (batch_size, horizon, target_dim)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_feature_names: list[str],
        input_feature_mode: InputFeatureMode = "explicit",
        input_feature_names: Optional[list[str]] = None,
        train_val_test_ratio: tuple[int, int, int] = (6, 2, 2),
        lookback: int = 12,
        horizon: int = 3,
        ahead: int = 0,
        normalize_x: bool = True,
        normalize_y: bool = True,
        time_col: str = "Year/Month",
        city_col: Optional[str] = "City",
        city: Optional[str] = None,
        fillna_method: Literal["zero", "ffill", "mean", "drop"] = "zero",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_shuffle: bool = True,
    ) -> None:
        super().__init__()

        self.df = df.copy()
        self.target_feature_names = target_feature_names
        self.input_feature_mode = input_feature_mode
        self.input_feature_names = input_feature_names

        self.train_val_test_ratio = train_val_test_ratio
        self.lookback = lookback
        self.horizon = horizon
        self.ahead = ahead

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

        self.time_col = time_col
        self.city_col = city_col
        self.city = city
        self.fillna_method = fillna_method

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_shuffle = train_shuffle

        self.bundle: Optional[ForecastDataBundle] = None

    # --------------------------------------------------------
    # Lightning API
    # --------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self.bundle is None:
            self.bundle = self._build_bundle()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bundle.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bundle.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bundle.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    # --------------------------------------------------------
    # Core build logic
    # --------------------------------------------------------

    def _build_bundle(self) -> ForecastDataBundle:
        df = self._prepare_dataframe(self.df)

        input_feature_names = self._resolve_input_feature_names(df)

        X_all = df[input_feature_names].to_numpy(dtype=np.float32)
        y_all = df[self.target_feature_names].to_numpy(dtype=np.float32)
        time_index = df[self.time_col].tolist()

        train_slice, val_slice, test_slice = self._make_time_splits(len(df))

        X_train_raw = X_all[train_slice]
        y_train_raw = y_all[train_slice]

        X_val_raw = X_all[val_slice]
        y_val_raw = y_all[val_slice]

        X_test_raw = X_all[test_slice]
        y_test_raw = y_all[test_slice]

        x_scaler = None
        y_scaler = None

        if self.normalize_x:
            x_scaler = SimpleStandardScaler().fit(X_train_raw)
            X_train = x_scaler.transform(X_train_raw)
            X_val = x_scaler.transform(X_val_raw)
            X_test = x_scaler.transform(X_test_raw)
        else:
            X_train = X_train_raw
            X_val = X_val_raw
            X_test = X_test_raw

        if self.normalize_y:
            y_scaler = SimpleStandardScaler().fit(y_train_raw)
            y_train = y_scaler.transform(y_train_raw)
            y_val = y_scaler.transform(y_val_raw)
            y_test = y_scaler.transform(y_test_raw)
        else:
            y_train = y_train_raw
            y_val = y_val_raw
            y_test = y_test_raw

        train_dataset = TimeSeriesWindowDataset(
            X_seq=X_train,
            y_seq=y_train,
            lookback=self.lookback,
            horizon=self.horizon,
            ahead=self.ahead,
        )

        val_dataset = TimeSeriesWindowDataset(
            X_seq=X_val,
            y_seq=y_val,
            lookback=self.lookback,
            horizon=self.horizon,
            ahead=self.ahead,
        )

        test_dataset = TimeSeriesWindowDataset(
            X_seq=X_test,
            y_seq=y_test,
            lookback=self.lookback,
            horizon=self.horizon,
            ahead=self.ahead,
        )

        return ForecastDataBundle(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,

            input_feature_names=input_feature_names,
            target_feature_names=self.target_feature_names,

            train_time_index=time_index[train_slice],
            val_time_index=time_index[val_slice],
            test_time_index=time_index[test_slice],

            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.time_col not in df.columns:
            raise ValueError(f"time_col={self.time_col!r} not found in df.columns.")

        df = df.copy()

        if self.city is not None:
            if self.city_col is None:
                raise ValueError("city_col must be provided when city is not None.")

            if self.city_col not in df.columns:
                raise ValueError(f"city_col={self.city_col!r} not found in df.columns.")

            df = df[df[self.city_col] == self.city].copy()

        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        needed_cols = set(self.target_feature_names)

        if self.input_feature_mode == "explicit":
            if self.input_feature_names is None:
                raise ValueError(
                    "input_feature_names must be provided "
                    "when input_feature_mode='explicit'."
                )
            needed_cols.update(self.input_feature_names)

        missing_cols = [col for col in needed_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        df = self._fill_missing_values(df)

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            col
            for col in df.columns
            if col not in {self.time_col, self.city_col}
        ]

        if self.fillna_method == "zero":
            df[feature_cols] = df[feature_cols].fillna(0)

        elif self.fillna_method == "ffill":
            df[feature_cols] = df[feature_cols].ffill().bfill()

        elif self.fillna_method == "mean":
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

        elif self.fillna_method == "drop":
            df = df.dropna(subset=feature_cols).reset_index(drop=True)
        elif self.fillna_method == None:
            df =df 
        else:
            raise ValueError(
                "fillna_method must be one of: 'zero', 'ffill', 'mean', 'drop'."
            )

        return df

    def _resolve_input_feature_names(self, df: pd.DataFrame) -> list[str]:
        non_feature_cols = {self.time_col}

        if self.city_col is not None and self.city_col in df.columns:
            non_feature_cols.add(self.city_col)

        if self.input_feature_mode == "explicit":
            if self.input_feature_names is None:
                raise ValueError(
                    "input_feature_names must be provided "
                    "when input_feature_mode='explicit'."
                )
            return list(self.input_feature_names)

        all_feature_cols = [
            col
            for col in df.columns
            if col not in non_feature_cols
        ]

        if self.input_feature_mode == "all":
            return all_feature_cols

        if self.input_feature_mode == "exclude_targets":
            return [
                col
                for col in all_feature_cols
                if col not in self.target_feature_names
            ]

        raise ValueError(
            "input_feature_mode must be one of: "
            "'all', 'exclude_targets', 'explicit'."
        )

    def _make_time_splits(
        self,
        n_time: int,
    ) -> tuple[slice, slice, slice]:
        r_train, r_val, r_test = self.train_val_test_ratio

        if r_train <= 0 or r_val <= 0 or r_test <= 0:
            raise ValueError("train_val_test_ratio values must all be positive.")

        ratio_sum = r_train + r_val + r_test

        train_end = int(n_time * r_train / ratio_sum)
        val_end = int(n_time * (r_train + r_val) / ratio_sum)

        min_len = self.lookback + self.ahead + self.horizon

        if train_end < min_len:
            raise ValueError(
                f"Train split is too short. Need at least {min_len}, got {train_end}."
            )

        if val_end - train_end < min_len:
            raise ValueError(
                f"Val split is too short. Need at least {min_len}, "
                f"got {val_end - train_end}."
            )

        if n_time - val_end < min_len:
            raise ValueError(
                f"Test split is too short. Need at least {min_len}, "
                f"got {n_time - val_end}."
            )

        train_slice = slice(0, train_end)
        val_slice = slice(train_end, val_end)
        test_slice = slice(val_end, n_time)

        return train_slice, val_slice, test_slice
