"""
Lightning DataModule for city-by-city multi-target forecasting.
"""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from .builder import MultiTargetCityDatasetBuilder


class ForecastDataModule(L.LightningDataModule):
    """
    LightningDataModule for city-by-city multi-target forecasting.

    Expected dataset builder output
    -------------------------------
    train_input:  (N, lookback, input_dim)
    train_target: (N, horizon, target_dim)
    val_input:    (N, lookback, input_dim)
    val_target:   (N, horizon, target_dim)
    test_input:   (N, lookback, input_dim)
    test_target:  (N, horizon, target_dim)
    """

    def __init__(
        self,
        dataset_config,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_shuffle: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_shuffle = train_shuffle

        self.bundle = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if self.bundle is None:
            builder = MultiTargetCityDatasetBuilder(self.dataset_config)
            self.bundle = builder.build()

        if stage in (None, "fit"):
            self.train_dataset = TensorDataset(
                self.bundle.train_input.float(),
                self.bundle.train_target.float(),
            )
            self.val_dataset = TensorDataset(
                self.bundle.val_input.float(),
                self.bundle.val_target.float(),
            )

        if stage in (None, "test", "predict"):
            self.test_dataset = TensorDataset(
                self.bundle.test_input.float(),
                self.bundle.test_target.float(),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,  # 你的 dataset 层已经可选 shuffle_train 了
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )