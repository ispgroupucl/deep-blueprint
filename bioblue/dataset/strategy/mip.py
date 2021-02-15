from collections import deque
from pathlib import Path
from itertools import islice
import numpy as np
import logging

from . import Strategy

log = logging.getLogger(__name__)


class MIPStrategy(Strategy):
    def __init__(
        self, source_dtype: str = "image", dtype_prefix: str = "mip", size: int = 10
    ) -> None:
        self.source_dtype = source_dtype
        self.dtype = f"{dtype_prefix}-{source_dtype}"
        self.size = size

    def prepare_data(self, data_dir: Path) -> None:
        for directory in data_dir.iterdir():
            if not directory.is_dir():
                continue
            if (
                not (directory / self.source_dtype).exists()
                and not (directory / (self.source_dtype + ".npz")).exists()
            ):
                log.error(f"{directory} does not contain the correct input files")
                continue
            self._prepare_dir(directory)

    def _prepare_dir(self, data_dir: Path) -> None:
        source_is_dir = (data_dir / self.source_dtype).is_dir()
        if not source_is_dir:
            log.warning(
                "Since all samples are in the same file, some border issues might arise."
            )
        if (data_dir / (self.dtype + ".npz")).exists():
            log.warning(
                f"{data_dir / (self.dtype + '.npz')} already exists, skipping creation."
            )
            return
        if (data_dir / self.dtype).exists():
            log.warning(f"{data_dir / self.dtype} already exists, skipping creation.")
            return
        if source_is_dir:
            source_files = list((data_dir / self.source_dtype).iterdir())
        else:
            source_files = [data_dir / (self.source_dtype + ".npz")]

        for source_file in source_files:
            source_data = np.load(source_file)
            target_data = {}
            tmp_mip = deque(
                [source_data[x] for x in islice(source_data, self.size // 2)], self.size
            )
            for i, source_name in enumerate(source_data.files):
                new_i = min(i + self.size // 2, len(source_data.files) - 1)
                tmp_mip.append(source_data[source_data.files[new_i]])
                concat_mip = np.stack(tmp_mip, axis=0)
                target_data[source_name] = np.max(concat_mip, axis=0)

            if source_is_dir:
                target_file = source_file.parents[1] / self.dtype / source_file.name
                target_file.parent.mkdir(exist_ok=True)
            else:
                target_file = source_file.parent / (self.dtype + ".npz")

            np.savez_compressed(target_file, **target_data)

