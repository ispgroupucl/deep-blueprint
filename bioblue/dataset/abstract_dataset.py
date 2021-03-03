from abc import ABC, abstractmethod
from typing import List
from torch.utils.data import Dataset
from pathlib import Path


class BioblueDataset(ABC, Dataset):
    @abstractmethod
    def __init__(self, root_dir: Path, dtypes: List[str]):
        self.root_dir = root_dir
        self.dtypes = dtypes
