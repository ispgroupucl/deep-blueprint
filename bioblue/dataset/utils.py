from numpy.lib.function_base import interp
from bioblue.dataset.transform.pipelines import Compose
import collections
import zipfile
from functools import partial
from torch.utils.data import Dataset
from pathlib import Path
from cachetools import cached
from cachetools.keys import hashkey
from collections import namedtuple, defaultdict
from skimage.util import view_as_windows
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

    def __init__(
        self, root_dir, partition, dtypes, transforms=None, remove_start=0, remove_end=0
    ) -> None:
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
        length = 0
        for file in self.files:
            data = np.load(self.root_dir / self.main_dtype / file)
            length += len(data.files[self.remove_start : len(data) - self.remove_end])
        return length

    def initialize(self):
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
                    # log.debug(e)
                    continue
                for j in range(self.remove_start, len(data) - self.remove_end):
                    self.reverse_index[dtype].append(i)
                    self.array_index[dtype].append(j)
                self.data[dtype].append(data)
                self.original_shape[dtype].append(data[data.files[0]].shape)

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
    crop=None,
    stride=None,
    partition="train",
    split=1,
    output_name=None,
    file_suffix=".bmp",
    seg_suffix=".bmp",
):
    output_dir = Path("/home/vjoosdeterbe/projects/bio-blueprints/data/")
    output_name = dataset_name if output_name is None else output_name
    datasets_dir = Path("~/projects/bb-data").expanduser()
    dataset_dir = datasets_dir / dataset_name
    for name, (img_dir, seg_dir) in sample_dirs.items():
        print(name)
        log.info(f"processing {name}.")
        img_files = list(sorted((dataset_dir / img_dir).iterdir()))
        img_files = [f for f in img_files if f.suffix == file_suffix]
        # test_img = cv2.imread(str(img_files[0]), cv2.IMREAD_GRAYSCALE)
        # original_shape = test_img.shape
        # test_img = view_as_windows(test_img, crop, step=stride)
        split_img_files = np.array_split(img_files, split)
        start = 0
        out_img_dir = output_dir / output_name / partition / "image"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        if seg_dir is not None:
            out_seg_dir = output_dir / output_name / partition / "segmentation"
            out_seg_dir.mkdir(parents=True, exist_ok=True)
        for img_files in split_img_files:
            img_filename = out_img_dir / f"{name}_{start}-{start+len(img_files)-1}.npz"
            img_zipf = zipfile.ZipFile(
                img_filename,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            )
            if seg_dir is not None:
                seg_filename = (
                    out_seg_dir / f"{name}_{start}-{start+len(img_files)-1}.npz"
                )
                seg_zipf = zipfile.ZipFile(
                    seg_filename,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                    allowZip64=True,
                )
            start += len(img_files)
            for i, img_file in enumerate(tqdm(img_files)):
                suffix = img_file.stem.rsplit(prefix, 1)[1]
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                image = image.astype(np.uint8)
                if resize:
                    image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
                with img_zipf.open(f"arr_{i}.npy", "w", force_zip64=True) as fid:
                    np.lib.format.write_array(fid, image)

                if seg_dir is not None:
                    seg_files = list(
                        (dataset_dir / seg_dir).glob(f"*{suffix}{seg_suffix}")
                    )
                    print(dataset_dir / seg_dir, suffix)
                    assert len(seg_files) <= 1
                    if len(seg_files) != 1:
                        continue
                    seg_file = seg_files[0]
                    print(seg_file)
                    seg = cv2.imread(str(seg_file), cv2.IMREAD_GRAYSCALE)
                    if resize:
                        seg = cv2.resize(seg, resize, interpolation=cv2.INTER_NEAREST)
                    if np.max(seg) == 255:
                        seg[seg == np.max(seg)] = 1
                    with seg_zipf.open(f"arr_{i}.npy", "w", force_zip64=True) as fid:
                        np.lib.format.write_array(fid, seg)
            img_zipf.close()
            if seg_dir is not None:
                seg_zipf.close()
