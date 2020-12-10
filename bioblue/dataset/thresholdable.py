import os
from typing import Optional

import numpy as np

from bioblue.dataset.utils import NumpyDataset
from pathlib import Path
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .downloadable import DownloadableDataModule
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class ThresholdableDataModule(DownloadableDataModule):
    def __init__(
        self,
        data_dir="./data",
        directory="kidneys",
        batch_size=2,
        num_workers=1,
        threshold_mean=0.999,
        threshold_std=0.001,
    ) -> None:
        self.threshold_mean = threshold_mean
        self.threshold_std = threshold_std
        super().__init__(data_dir, directory, batch_size, num_workers)

    def setup(self, stage: Optional[str] = None):
        self.train = ThresholdableDataset(
            self.dir / "train", ["image"], self.threshold_mean, self.threshold_std
        )
        self.val = NumpyDataset(self.dir / "val", ["image", "segmentation"])
        self.test = NumpyDataset(self.dir / "test", ["image", "segmentation"])


class ThresholdableDataset(NumpyDataset):
    def __init__(self, root_dir, dtypes, mean=0.999, std=0.001):
        super().__init__(root_dir, dtypes)
        self.mean = mean
        self.std = std

    def __getitem__(self, index: int):
        result = super().__getitem__(index)
        img = result["image"]
        seg = np.zeros_like(img)  # .to(torch.long)
        rng = np.random.default_rng()
        threshold = np.clip(rng.normal(self.mean, self.std), 0, 1)
        p = np.quantile(img, threshold)
        seg[img >= p] = 1
        return dict(image=img, segmentation=seg)
