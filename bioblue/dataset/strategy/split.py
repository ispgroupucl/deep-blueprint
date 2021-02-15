from typing import Union
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
        end = (self.fold + 1) * fold_length if self.fold < self.folds - 1 else -1
        val_ds.files = val_ds.files[self.fold * fold_length : end]
        train_ds.files = list(set(train_ds.files) - set(val_ds.files))

        log.info(
            f"Split #{self.fold+1}/{self.folds} : "
            f"{len(train_ds)} samples for train and {len(val_ds)} for validation."
        )
