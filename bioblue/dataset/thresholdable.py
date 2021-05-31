import os
from typing import Optional

import numpy as np

from bioblue.dataset.utils import MultipleNumpyDataset, NumpyDataset
from pathlib import Path
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .downloadable import DownloadableDataModule
from torch.utils.data import Dataset
from hydra.utils import call

log = logging.getLogger(__name__)


class ThresholdableDataset(MultipleNumpyDataset):
    def __init__(
        self,
        root_dir,
        partition,
        dtypes,
        transforms=None,
        mean=0.999,
        std=0.001,
        rand_threshold: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(root_dir, partition, dtypes, transforms=transforms, **kwargs)
        self.rand_threshold = rand_threshold or dict(
            _target_="numpy.random.normal", loc=mean, scale=std
        )

    def __getitem__(self, index: int):
        result = super().__getitem__(index, do_transform=False)
        img = result["image"]
        seg = np.zeros_like(img)
        rng = np.random.default_rng()
        threshold = np.clip(call(self.rand_threshold), 0, 1)
        p = np.quantile(img, threshold)
        seg[img >= p] = 1
        result["segmentation"] = seg
        result["_title"] = f"thresh={threshold:.4f} quantile={p}"
        if self.transforms is not None:
            result = self.transforms(**result)
        return result


class GlobalThresholdableDataset(MultipleNumpyDataset):
    def __init__(
        self, root_dir, partition, dtypes, transforms=None, low=225, high=250, **kwargs,
    ) -> None:
        super().__init__(root_dir, partition, dtypes, transforms=transforms, **kwargs)
        self.low = low
        self.high = high

    def __getitem__(self, index: int):
        result = super().__getitem__(index, do_transform=False)
        img = result["image"]
        if "segmentation" not in result:
            seg = np.zeros_like(img)
            rng: np.random.Generator = np.random.default_rng()
            threshold = rng.integers(self.low, self.high, endpoint=True)
            seg[img >= threshold] = 1
            result["segmentation"] = seg
        else:
            log.debug(f"{index} already contains segmentation")
        if self.transforms is not None:
            result = self.transforms(**result)
        return result


# class ThresholdableDataModule(DownloadableDataModule):
#     def __init__(
#         self,
#         data_dir="./data",
#         directory="kidneys",
#         batch_size=2,
#         num_workers=1,
#         threshold_mean=0.999,
#         threshold_std=0.001,
#     ) -> None:
#         self.threshold_mean = threshold_mean
#         self.threshold_std = threshold_std
#         super().__init__(data_dir, directory, batch_size, num_workers)

#     def setup(self, stage: Optional[str] = None):
#         self.train = ThresholdableDataset(
#             self.dir / "train", ["image"], self.threshold_mean, self.threshold_std
#         )
#         self.val = NumpyDataset(self.dir / "val", ["image", "segmentation"])
#         self.test = NumpyDataset(self.dir / "test", ["image", "segmentation"])


# class ThresholdableDataset(NumpyDataset):
#     def __init__(self, root_dir, dtypes, mean=0.999, std=0.001):
#         super().__init__(root_dir, dtypes)
#         self.mean = mean
#         self.std = std

#     def __getitem__(self, index: int):
#         result = super().__getitem__(index)
#         img = result["image"]
#         seg = np.zeros_like(img)  # .to(torch.long)
#         rng = np.random.default_rng()
#         threshold = np.clip(rng.normal(self.mean, self.std), 0, 1)
#         p = np.quantile(img, threshold)
#         seg[img >= p] = 1
#         return dict(image=img, segmentation=seg)
