import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from numpy.lib.arraysetops import isin
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from bioblue.dataset import PrepareStrategy, SetupStrategy
from hydra.utils import instantiate

log = logging.getLogger(__name__)


class BioblueDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        train_dataset: Optional[dict] = None,
        val_dataset: Optional[dict] = None,
        test_dataset: Optional[dict] = None,
        strategies: Optional[Dict[str, Union[PrepareStrategy, SetupStrategy]]] = None,
        batch_size: int = 2,
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) / dataset_name
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset

        # print(self.data_dir)

        if strategies is not None:
            self.strategies = {}
            for name, strat in strategies.items():
                strategy = instantiate(strat)
                if not isinstance(strategy, (PrepareStrategy, SetupStrategy)):
                    raise TypeError(
                        f"{name} does not implement one of the Strategy classes"
                    )
                self.strategies[name] = strategy
        else:
            self.strategies = {}
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        for name, strategy in self.strategies.items():
            if isinstance(strategy, PrepareStrategy):
                log.info(f"Data preparation : {name}")
                strategy.prepare_data(self.data_dir)

    def setup(self, stage=None):
        self.train_ds = instantiate(
            self.train_ds, root_dir=self.data_dir, _recursive_=False
        )
        self.val_ds = instantiate(
            self.val_ds, root_dir=self.data_dir, _recursive_=False
        )
        self.test_ds = instantiate(
            self.test_ds, root_dir=self.data_dir, _recursive_=False
        )

        for name, strategy in self.strategies.items():
            if isinstance(strategy, SetupStrategy):
                log.info(f"Dataset setup: {name}")
                strategy.setup(self.train_ds, self.val_ds, self.test_ds)

        log.info(f"train size: {len(self.train_ds)}; val size: {len(self.val_ds)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
        )
