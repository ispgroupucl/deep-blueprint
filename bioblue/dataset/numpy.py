from contextlib import AbstractContextManager
import zipfile
from bioblue.dataset.transform.pipelines import Compose
import collections
from functools import partial
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
import logging
import numpy as np
from hydra.utils import call, instantiate
import json

log = logging.getLogger(__name__)


class MultipleNumpyDataset(Dataset):
    """ A dataset containing a directory per type (image, segmentation)

        Every type directory contains an .npz file per sample (or volume), the 
        number of files per directory must be the same, as must be the names of
        the files.

        Args:
            root_dir: directory containing the set directories.
            partition: subdirectory inside the root_dir (train, test or val).
            dtypes: the types that must be existing directories.
            transforms: a callable, a dict with _target_ or a list of dicts with
                _target_'s the list will be passed through a custom Compose method.
    
    """

    def __init__(
        self, root_dir, partition, dtypes, transforms=None, remove_start=0, remove_end=0
    ) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
            if "_target_" in transforms:
                transforms = instantiate(transforms, _recursive_=True)
            else:
                transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)
        self.transforms = transforms
        self.root_dir = Path(root_dir) / partition
        self.dtypes = dtypes
        self.data = None
        self.files = []
        self.reverse_index = None
        self.array_index = None
        self.original_shape = None
        self.remove_start = remove_start
        self.remove_end = remove_end
        self.main_dtype = dtypes[0]
        all_len = []
        for dtype in dtypes:
            dtype_len = 0
            for file in sorted((self.root_dir / dtype).iterdir()):
                if dtype == self.main_dtype:
                    self.files.append(file.name)
                data = np.load(file)
                dtype_len += len(
                    data.files[self.remove_start : len(data) - self.remove_end]
                )
            all_len.append(dtype_len)

        log.debug(f"{self.files}")
        if len(np.unique(all_len)) != 1:
            log.warning("Unequal number of images")

    def __len__(self) -> int:
        log.debug("called length")
        length = 0
        for file in self.files:
            data = np.load(self.root_dir / self.main_dtype / file)
            length += len(data.files[self.remove_start : len(data) - self.remove_end])
            data.close()
        return length

    def initialize(self):
        log.debug("called initialize")
        self.reverse_index = defaultdict(list)
        self.array_index = defaultdict(list)
        self.data = defaultdict(list)
        self.original_shape = defaultdict(list)
        for dtype in self.dtypes:
            for i, filename in enumerate(self.files):
                file = self.root_dir / dtype / filename
                try:
                    data = np.load(file)
                except Exception as e:
                    log.warning(e)
                    continue
                for j in range(self.remove_start, len(data) - self.remove_end):
                    self.reverse_index[dtype].append(i)
                    self.array_index[dtype].append(j)
                self.data[dtype].append(data)
                self.original_shape[dtype].append(data[data.files[0]].shape)
        log.debug("end initialize")

    def reset(self):
        for _, data in self.data.items():
            for item in data:
                item.close()

        self.data = None
        self.reverse_index = None
        self.array_index = None

    def __getitem__(self, index: int, do_transform=True):
        if self.data is None:
            self.initialize()

        sample = {}
        log.debug(
            "%s, idx: %d; file: %d; array: %d",
            self.main_dtype,
            index,
            self.reverse_index["image"][index],
            self.array_index["image"][index],
        )
        file_index = self.reverse_index[self.main_dtype][index]
        array_index = self.array_index[self.main_dtype][index]
        array_name = self.data[self.main_dtype][file_index].files[array_index]
        for dtype in self.dtypes:
            try:
                sample[dtype] = self.data[dtype][file_index][array_name]
            except Exception as e:
                log.debug(f"{dtype} {file_index} {array_name}")
                log.debug(f"{e}")

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)
        return sample


class SummaryNumpyDataset(MultipleNumpyDataset):
    def __init__(
        self, root_dir, partition, dtypes, transforms=None, remove_start=0, remove_end=0
    ) -> None:
        super(MultipleNumpyDataset, self).__init__()
        if isinstance(transforms, collections.Mapping):
            if "_target_" in transforms:
                transforms = instantiate(transforms, _recursive_=True)
            else:
                transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)
        self.transforms = transforms
        self.root_dir = Path(root_dir) / partition
        self.dtypes = dtypes
        self.data = None
        self.files = []
        self.reverse_index = None
        self.array_index = None
        self.original_shape = None
        self.remove_start = remove_start
        self.remove_end = remove_end
        self.main_dtype = dtypes[0]
        all_len = []
        with open(root_dir / "summary.json") as summary_fp:
            summary = json.load(summary_fp)
            summary = summary[summary["latest"]]
        for dtype in dtypes:
            dtype_len = 0
            for file in sorted(summary[partition][dtype]):
                file = root_dir / file
                if dtype == self.main_dtype:
                    self.files.append(file.name)
                data = np.load(file)
                dtype_len += len(
                    data.files[self.remove_start : len(data) - self.remove_end]
                )
            all_len.append(dtype_len)

        log.debug(f"{self.files}")
        if len(np.unique(all_len)) != 1:
            log.warning("Unequal number of images")

    def initialize(self):
        log.debug("called initialize")
        self.reverse_index = defaultdict(list)
        self.array_index = defaultdict(list)
        # self.data = defaultdict(list)
        self.original_shape = defaultdict(list)
        for dtype in self.dtypes:
            for i, filename in enumerate(self.files):
                file = self.root_dir / dtype / filename
                try:
                    data = np.load(file)
                except Exception as e:
                    log.warning(e)
                    continue
                for j in range(self.remove_start, len(data) - self.remove_end):
                    self.reverse_index[dtype].append(i)
                    self.array_index[dtype].append(j)
                # self.data[dtype].append(data)
                if i == 0:
                    shape = data[data.files[0]].shape
                self.original_shape[dtype].append(shape)
                data.close()
        log.debug("end initialize")

    def reset(self):
        self.data = None
        self.reverse_index = None
        self.array_index = None

    def __getitem__(self, index: int, do_transform=True):
        log.debug("start getitem")
        if self.reverse_index is None:
            self.initialize()

        sample = {}
        log.debug(
            "%s, idx: %d; file: %d; array: %d",
            self.main_dtype,
            index,
            self.reverse_index["image"][index],
            self.array_index["image"][index],
        )
        file_index = self.reverse_index[self.main_dtype][index]
        array_index = self.array_index[self.main_dtype][index]
        for dtype in self.dtypes:
            try:
                file = self.root_dir / dtype / self.files[file_index]
                data = np.load(file)
                array_name = data.files[array_index]
                sample[dtype] = data[array_name]
                log.debug(f"{sample[dtype].shape}")
            except Exception as e:
                log.debug(f"{dtype} {file_index} {array_name}")
                log.debug(f"{e}")
        if self.transforms is not None and do_transform:
            sample = self.transforms(sample)

        log.debug("end getitem")
        return sample


class NpzWriter(AbstractContextManager):
    def __init__(self, filename):
        self.zf = zipfile.ZipFile(
            filename, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True,
        )
        self.i = 0

    def add(self, array):
        with self.zf.open(f"arr_{self.i}.npy", "w", force_zip64=True) as fid:
            np.lib.format.write_array(fid, array)
        self.i += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.zf.close()

    def close(self):
        self.zf.close()
