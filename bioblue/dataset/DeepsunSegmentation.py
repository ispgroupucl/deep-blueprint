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
import os
from hydra.utils import call, instantiate
import skimage.io as io


log = logging.getLogger(__name__)

class DeepsunSegmentationDataset(Dataset):
    """ A dataset containing a directory per type (image, segmentation1, segmetnation2...)

        Every type directory contains an image/target file per sample , the 
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
        self, root_dir, partition, dtypes, transforms=None) -> None:
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

        self.main_dtype = dtypes[0]
        self.target_types = dtypes[1:]

        self.files = []
        self.masks_lists = { t: sorted((self.root_dir / t).iterdir()) for t in self.target_types}
        
        for i, file in enumerate(sorted((self.root_dir / self.main_dtype).iterdir())):
            cur = {}
            cur[self.main_dtype] = file
            cur['name'] = os.path.basename(file)
            for t in self.target_types:
                cur[t] = self.masks_lists[t][i]

            self.files.append(cur)



    def __len__(self) -> int:
        return len(self.files)


    def __getitem__(self, index: int, do_transform=True):

        sample = {} # dictionary with 'image', "segmentation" entries

        img_name =  self.files[index]["image"]
        # try:
        sample["image"] = (io.imread(img_name)).astype(float) # load image from directory with skimage
        # print(sample["image"].dtype)
        # except Exception as e:
        #     log.debug(f"{self.main_dtype} {index} {img_name}")
        #     log.debug(f"{e}")
        
        # pick one segmentation according to distribution
        segmentation_type = "segmentation1"
        # try:
        seg_file = self.files[index][segmentation_type]
        sample["segmentation"] = (io.imread(seg_file)).astype(float) # load corresponding segmentation mask

        print(sample["segmentation"].dtype)
        # except Exception as e:
        #     log.debug(f"{self.main_dtype} {index} {segmentation_type}")
        #     log.debug(f"{e}")

        
        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)

        return sample