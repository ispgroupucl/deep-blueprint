from torch.utils.data import Dataset
from pathlib import Path
from cachetools import cached
from cachetools.keys import hashkey
from collections import namedtuple
import nibabel
import numpy as np


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


class NumpyDataset(Dataset):
    def __init__(self, root_dir, dtypes):
        self.dtypes = dtypes
        self.files = {}
        all_len = []
        for dtype in dtypes:
            data = np.load(root_dir / (dtype + ".npz"))
            self.files[dtype] = data
            all_len.append(len(data.files))

        assert len(np.unique(all_len)) == 1, f"unequal number of images"

        self.length = all_len[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {
            dtype: self.files[dtype][self.files[dtype].files[index]]
            for dtype in self.dtypes
        }

