from bioblue.dataset.utils import DirectoryDataset
from pathlib import Path
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import numpy as np
import nibabel


class DVNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 2,
        directory: str = "deepvesselnet",
        dtypes=("raw", "seg"),
        splits=(0.9, 0.05, 0.05),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dir = directory
        self.batch_size = batch_size
        self.dtypes = dtypes
        self.splits = splits

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = DirectoryDataset(
            root_dir=Path(self.data_dir) / self.dir, dtypes=self.dtypes
        )
        if self.splits[0] < 1:
            splits = [int(x) for x in len(dataset) * np.array(self.splits)]
            splits[0] += len(dataset) - sum(splits)
        else:
            splits = self.splits

        datasets = random_split(dataset, lengths=splits)
        self.train = datasets[0]
        self.val = datasets[1]
        self.test = datasets[2]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

