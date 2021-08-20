from bioblue.dataset.numpy import NpzWriter
from pathlib import Path

import numpy as np
from . import Strategy
from bioblue import fibers
import logging

log = logging.getLogger(__name__)


class CropStrategy(Strategy):
    def __init__(self, size=None, step=None, dtype="image", partition="train"):
        self.size = size
        self.step = step
        self.partition = partition
        self.dtype = dtype

    def prepare_data(self, data_dir: Path) -> None:
        for np_file in (data_dir / self.partition / self.dtype).iterdir():
            log.debug(f"processing {np_file.name}")
            sample = np.load(np_file)
            slice = sample[sample.files[0]]
            number = [
                (shape // size) + 1 for shape, size in zip(slice.shape, self.size)
            ]
            if self.step is None:
                step = [
                    (shape - size) / (n - 1)
                    for shape, size, n in zip(slice.shape, self.size, number)
                ]
            else:
                step = self.step
            log.debug(f"{slice.shape} {number} {step} {self.size}")
            zf_dict = {}
            if (
                data_dir
                / self.partition
                / self.dtype
                / (np_file.stem + f"_{0}_{0}.npz")
            ).exists():
                continue
            for slicename in sample:
                slice = sample[slicename]
                log.debug(f"")
                for i in range(number[0]):
                    for j in range(number[1]):
                        filename = (
                            data_dir
                            / self.partition
                            / self.dtype
                            / (np_file.stem + f"_{i}_{j}.npz")
                        )
                        if (i, j) not in zf_dict:
                            zf_dict[(i, j)] = NpzWriter(filename)
                        zf = zf_dict[(i, j)]
                        log.debug(
                            f"start {i * step[0], j * step[1]} "
                            f"stop {i * step[0]+self.size[0], j * step[1] + self.size[1]}"
                        )
                        crop = slice[
                            round(i * step[0]) : round(i * step[0]) + self.size[0],
                            round(j * step[1]) : round(j * step[1]) + self.size[1],
                        ]
                        assert crop.shape == self.size
                        zf.add(crop)
            np_file.unlink()
