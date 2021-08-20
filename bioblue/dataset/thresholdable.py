import os
from typing import Optional

import numpy as np

from bioblue.dataset.numpy import MultipleNumpyDataset
import logging
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
