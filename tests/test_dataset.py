from bioblue import dataset
import pytest
from hypothesis import given
import hypothesis.strategies as hs
import torch
import numpy as np
from pathlib import Path

import bioblue as bb


def test_md5_dir(tmp_path):
    dataset_name = "test_dataset"
    dm = dataset.DownloadableDataModule(data_dir=tmp_path, directory=dataset_name)
    dm.prepare_data()
    directory = tmp_path / dataset_name
    run1 = dataset.md5_dir(directory)
    run2 = dataset.md5_dir(directory)
    assert run1 == run2


@pytest.mark.parametrize(
    "dataset_name", ["test_dataset", pytest.param("kidneys", marks=pytest.mark.slow)],
)
def test_downloadable_datamodule(tmp_path, dataset_name):
    print(tmp_path)
    dm = dataset.DownloadableDataModule(data_dir=tmp_path, directory=dataset_name)
    dm.prepare_data()
    dm.setup()
    assert "image" in dm.train[0]
    dm.prepare_data()
    dm.setup()
    assert "image" in dm.train[0]


# def test_failing_downloadable_datamodule(tmp_path):
