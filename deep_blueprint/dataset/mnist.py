from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import collections
from functools import partial
from ..dataset.transform.pipelines import Compose
from hydra.utils import call, instantiate


class MNistDataset(Dataset):
    def __init__(self, root_dir, partition, dtypes, transforms=None):
        super().__init__()
        self.data_dir = root_dir
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if isinstance(transforms, collections.abc.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.abc.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        self.transforms = transforms
        self.partition = partition
        self.dtypes = dtypes
        # print("lol", self.transforms)
        self.prepare_data()


    def prepare_data(self):
        # download
        istrain = self.partition == 'train'
        if istrain:
            self.ds = MNIST(self.data_dir, train=True, download=True)
        else:
            self.ds = MNIST(self.data_dir, train=False, download=True)


    
    def __len__(self) -> int:
        return len(self.ds)


    def __getitem__(self, index: int, do_transform=True):
        
        sample_img, sample_class = self.ds.__getitem__(index)

        # print(sample_img.width,'x',sample_img.height,'-> ', sample_class)

        sample = {'image': np.array(sample_img), 'class': np.array([sample_class])}

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)

        return sample