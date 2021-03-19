from bioblue.dataset.transform.pipelines import Compose
import collections
from functools import partial
from torch.utils.data import Dataset
from pathlib import Path
from cachetools import cached
from cachetools.keys import hashkey
from collections import namedtuple, defaultdict
import nibabel
import logging
import numpy as np
import cv2
from hydra.utils import call, instantiate
from tqdm.auto import tqdm

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

    def __init__(self, root_dir, partition, dtypes, transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
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
        for dtype in self.dtypes:
            file_index = self.reverse_index[dtype][index]
            array_index = self.array_index[dtype][index]
            array_name = self.data[dtype][file_index].files[array_index]
            sample[dtype] = self.data[dtype][file_index][array_name]

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)
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


def preprocess(
    dataset_name,
    sample_dirs,
    prefix="_",
    resize=(512, 512),
    partition="train",
    split=1,
    output_name=None,
    file_suffix=".bmp",
):
    output_dir = Path("/home/vjoosdeterbe/projects/bio-blueprints/data/")
    output_name = dataset_name if output_name is None else output_name
    datasets_dir = Path("~/projects/bb-data").expanduser()
    dataset_dir = datasets_dir / dataset_name
    for name, (img_dir, seg_dir) in sample_dirs.items():
        print(name)
        log.info(f"processing {name}.")
        output_image = []
        output_seg = []
        for i, img_file in enumerate(tqdm(sorted((dataset_dir / img_dir).iterdir()))):
            if not img_file.suffix == file_suffix:
                continue
            suffix = img_file.name.rsplit(prefix, 1)[1]
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.uint8)
            if resize:
                image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
            if seg_dir is not None:
                seg_files = list((dataset_dir / seg_dir).glob(f"*{suffix}"))
                assert len(seg_files) == 1
                seg_file = seg_files[0]
                seg = cv2.imread(str(seg_file), cv2.IMREAD_GRAYSCALE)
                if resize:
                    seg = cv2.resize(seg, resize, interpolation=cv2.INTER_NEAREST)
                if np.max(seg) != 1:
                    seg[seg == np.max(seg)] = 1
                output_seg.append(seg)
            output_image.append(image)
        output_segs = np.array_split(output_seg, split)
        output_images = np.array_split(output_image, split)
        if seg_dir is None:
            image_dir = output_dir / output_name / partition / "image"
        else:
            image_dir = output_dir / output_name / partition / "image"
            seg_dir = output_dir / output_name / partition / "segmentation"
            seg_dir.mkdir(parents=True, exist_ok=True)
            start = 0
            for output_seg in output_segs:
                np.savez_compressed(
                    seg_dir / f"{name}_{start}-{start+len(output_seg)-1}.npz",
                    *output_seg,
                )
                start += len(output_seg)
        image_dir.mkdir(parents=True, exist_ok=True)
        start = 0
        for output_image in output_images:
            np.savez_compressed(
                image_dir / f"{name}_{start}-{start+len(output_image)-1}.npz",
                *output_image,
            )
            start += len(output_image)
