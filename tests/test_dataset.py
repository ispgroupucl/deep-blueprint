from bioblue import dataset
import pytest
from hypothesis import given
import hypothesis.strategies as hs
import torch
import numpy as np
from pathlib import Path

import bioblue as bb


# @pytest.fixture(scope="session")
# def dataset_path(tmp_path_factory):
#     tmp_path = tmp_path_factory.mktemp("data-")
#     dataset_name = "test_dataset"
#     dm = dataset.DownloadableDataModule(data_dir=tmp_path, directory=dataset_name)
#     dm.prepare_data()
#     return tmp_path / dataset_name


# def test_md5_dir(dataset_path):
#     run1 = dataset.md5_dir(dataset_path)
#     run2 = dataset.md5_dir(dataset_path)
#     assert run1 == run2


# def test_thresholdable_dm(dataset_path):
#     dm = dataset.ThresholdableDataModule(
#         data_dir=dataset_path.parent, directory=dataset_path.name
#     )
#     dm.prepare_data()
#     dm.setup()
#     segms_sum = []
#     for _ in range(10):
#         segms_sum.append(np.sum(dm.train[0]["segmentation"]))

#     print(segms_sum)
#     assert len(set(segms_sum)) > 1

# @pytest.mark.parametrize(
#     "dataset_name", ["test_dataset"]
# )
# def test_datamodule(tmp_path, dataset_name):
#     dm = dataset.BioblueDataModule(data_dir=tmp_path, dataset_name=dataset_name)


# @pytest.mark.parametrize(
#     "dataset_name", ["test_dataset", pytest.param("kidneys", marks=pytest.mark.slow)],
# )
# def test_downloadable_datamodule(tmp_path, dataset_name):
#     print(tmp_path)
#     dm = dataset.DownloadableDataModule(data_dir=tmp_path, directory=dataset_name)
#     dm.prepare_data()
#     dm.setup()
#     assert "image" in dm.train[0]
#     dm.prepare_data()
#     dm.setup()
#     assert "image" in dm.train[0]


# def test_failing_downloadable_datamodule(tmp_path):
