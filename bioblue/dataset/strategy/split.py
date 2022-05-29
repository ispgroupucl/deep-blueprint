from typing import Dict, List, Optional, Union
import numpy as np
import logging
from pathlib import Path

from . import SetupStrategy, PrepareStrategy

log = logging.getLogger(__name__)


class SplitStrategy(PrepareStrategy):
    def __init__(
        self,
        fold: int,
        folds: int = 5,
        input_partition="train",
        train: Optional[float] = 0.8,
        val: Optional[float] = None,
        test: Optional[float] = 0.2,
    ) -> None:
        assert 0 <= fold < folds
        self.fold = fold
        self.folds = folds
        self.train_size = train
        self.val_size = val
        self.test_size = test

    def write_files(
        self, data_dir: Path, latest_files: Dict[str, Dict[str, List[Path]]]
    ) -> Dict[str, Dict[str, List[Path]]]:
        assert latest_files is not None
        return super().write_files(data_dir, latest_files)


class KFoldStrategy(SetupStrategy):
    def __init__(self, fold, folds=5) -> None:
        assert 0 <= fold < folds
        self.fold = fold
        self.folds = folds

    def setup(self, train_ds, val_ds, test_ds) -> None:
        assert (
            train_ds.root_dir == val_ds.root_dir
        ), "When using k-fold cross-validation, make sure to point train and val to the same root directory"

        length = len(train_ds.files)
        fold_length = length // self.folds
        end = (
            (self.fold + 1) * fold_length
            if self.fold < self.folds - 1
            else len(train_ds.files)
        )
        val_ds.files = train_ds.files[self.fold * fold_length : end]
        train_ds.files = list(set(train_ds.files) - set(val_ds.files))
        log.info(
            f"Split #{self.fold+1}/{self.folds} : "
            f"{len(train_ds)} samples for train and {len(val_ds)} for validation."
            f" (starting from {self.fold*fold_length} to {end})"
        )
        log.debug(f"training set containing {train_ds.files}")


class NamedKFoldStrategy(SetupStrategy):
    def __init__(self, fold, folds: List[list]) -> None:
        assert fold < len(folds)
        super().__init__()
        self.fold = fold
        self.folds = folds

    def setup(self, train_ds, val_ds, test_ds) -> None:
        assert (
            train_ds.root_dir == val_ds.root_dir
        ), "When using k-fold, train and val should be the same directory"

        not_used = []
        for file in sum(self.folds, []):
            if file not in train_ds.files and file not in val_ds.files:
                not_used.append(file)

        if len(not_used) > 0:
            log.warning(f"Some files will not be in validation ({', '.join(not_used)})")

        val_ds.files = self.folds[self.fold]
        train_ds.files = list(set(train_ds.files) - set(val_ds.files))
        log.info(
            f"Split #{self.fold+1}/{len(self.folds)} : "
            f"{len(train_ds)} samples for train and {len(val_ds)} for validation."
            f"\nvalidation set containing {val_ds.files}"
            f"\ntraining set containing {train_ds.files}"
        )

