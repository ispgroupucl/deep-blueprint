from pathlib import Path
from typing import List, Optional, Type
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from bioblue.dataset import Strategy, BioblueDataset
from hydra.utils import instantiate


class BioblueDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        train_dataset: Optional[dict] = None,
        val_dataset: Optional[dict] = None,
        test_dataset: Optional[dict] = None,
        strategies: Optional[List[Strategy]] = (),
        batch_size: int = 2,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset
        self.strategies = [instantiate(strat) for strat in strategies]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        for strategy in self.strategies:
            strategy.prepare_data(self.data_dir)

    def setup(self, stage=None):
        self.train_ds = instantiate(
            self.train_ds
        )  # , root_dir=self.data_dir / "train")
        self.val_ds = instantiate(self.val_ds)  # , root_dir=self.data_dir / "val")
        self.test_ds = instantiate(self.test_ds)  # , root_dir=self.data_dir / "test")

        for strategy in self.strategies:
            strategy.setup(self.train_ds, self.val_ds, self.test_ds)

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
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
