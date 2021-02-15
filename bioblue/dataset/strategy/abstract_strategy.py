from abc import ABC, abstractmethod
from typing import Sequence, Union
from pathlib import Path
from pytorch_lightning import LightningDataModule


class Strategy(ABC):
    def prepare_data(self, data_dir: Path) -> None:
        pass

    def setup(self, train_ds, val_ds, test_ds) -> None:
        pass
