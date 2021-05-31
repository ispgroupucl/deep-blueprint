from typing import List, Union
import numpy as np
import logging
from pathlib import Path

from . import Strategy

log = logging.getLogger(__name__)


class KFoldStrategy(Strategy):
    def __init__(self, fold, folds=5) -> None:
        assert fold < folds
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
            f"\n training set containing {train_ds.files}"
        )


class NamedKFoldStrategy(Strategy):
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

