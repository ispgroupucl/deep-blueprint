from torch.utils.data import Dataset
from pathlib import Path
from cachetools import cached
from cachetools.keys import hashkey
from collections import namedtuple, defaultdict
import nibabel
import logging
import numpy as np

log = logging.getLogger(__name__)


def read_sample(samplefiles=None, **kwargs):
    if samplefiles is None:
        samplefiles = kwargs
    else:
        samplefiles.update(kwargs)

    result = {}
    for part in samplefiles:
        filepath = Path(samplefiles[part])
        part_type = filepath.suffixes

        if ".nii" in part_type:
            result[part] = nibabel.load(filepath).get_fdata()
        else:
            raise ValueError(
                f"{part} is either not a valid type or not yet implemented."
            )

    return result


class DirectoryDataset(Dataset):
    def __init__(self, root_dir, dtypes):
        self.dtypes = dtypes
        self.fnames = {}
        for i, dtype in enumerate(dtypes):
            fnames = (Path(root_dir) / dtype).iterdir()
            self.fnames[dtype] = sorted(fnames)

        all_len = [len(elem) for elem in self.fnames.values()]

        # Check if all directories have the same number of files
        assert (
            len(np.unique(all_len)) == 1
        ), f"The number of images is not equal for the different inputs:\n {dtypes}\n {all_len}"

        self.length = all_len[0]
        self.fname_samples = [
            dict(zip(self.fnames, i)) for i in zip(*self.fnames.values())
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        result = read_sample(self.fname_samples[idx])
        return result


class MultipleNumpyDataset(Dataset):
    def __init__(self, root_dir, dtypes, transforms=None) -> None:
        super().__init__()
        self.transforms = transforms
        self.root_dir = Path(root_dir)
        self.dtypes = dtypes
        self.data = None
        self.files = []
        self.reverse_index = None
        self.array_index = None
        all_len = []
        for dtype in dtypes:
            dtype_len = 0
            for file in sorted((self.root_dir / dtype).iterdir()):
                if dtype == dtypes[0]:
                    self.files.append(file.name)
                data = np.load(file)
                dtype_len += len(data.files)
            all_len.append(dtype_len)

        assert len(np.unique(all_len)) == 1, "unequal number of images"

    def __len__(self) -> int:
        length = 0
        for file in self.files:
            data = np.load(self.root_dir / self.dtypes[0] / file)
            length += len(data.files)
        return length

    def initialize(self):
        self.reverse_index = defaultdict(list)
        self.array_index = defaultdict(list)
        self.data = defaultdict(list)
        for dtype in self.dtypes:
            for i, filename in enumerate(self.files):
                file = self.root_dir / dtype / filename
                data = np.load(file)
                for j in range(len(data)):
                    self.reverse_index[dtype].append(i)
                    self.array_index[dtype].append(j)
                self.data[dtype].append(data)

    def __getitem__(self, index: int):
        if self.data is None:
            self.initialize()

        sample = {}
        for dtype in self.dtypes:
            file_index = self.reverse_index[dtype][index]
            array_index = self.array_index[dtype][index]
            array_name = self.data[dtype][file_index].files[array_index]
            sample[dtype] = self.data[dtype][file_index][array_name]

        return sample


class NumpyDataset(Dataset):
    def __init__(self, root_dir, dtypes):
        self.root_dir = root_dir
        self.dtypes = dtypes
        self.files = None
        all_len = []
        for dtype in dtypes:
            data = np.load(root_dir / (dtype + ".npz"))
            # self.files[dtype] = data
            all_len.append(len(data.files))

        assert len(np.unique(all_len)) == 1, f"unequal number of images"

        self.length = all_len[0]

    def __len__(self) -> int:
        return self.length

    def initialize(self):
        self.files = {}
        for dtype in self.dtypes:
            data = np.load(self.root_dir / (dtype + ".npz"))
            self.files[dtype] = data

    def __getitem__(self, index: int):
        # initialize file pointers inside each individual worker
        if self.files is None:
            self.initialize()
        return {
            dtype: self.files[dtype][self.files[dtype].files[index]]
            for dtype in self.dtypes
        }

