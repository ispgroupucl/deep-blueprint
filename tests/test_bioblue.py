import pytest

from bioblue import __version__
import bioblue as bb


def test_version():
    assert __version__ == "0.1.0"


def test_directorydataset():
    dir_ds = bb.dataset.DirectoryDataset(
        root_dir="./data/deepvesselnet", dtypes=["raw", "seg"]
    )
    assert len(dir_ds) == 136
    first_element = dir_ds[0]
    assert len(first_element) == 2
    assert "raw" in first_element and "seg" in first_element
    assert len(first_element["raw"].shape) == 3
    assert first_element["raw"].shape == first_element["seg"].shape


dl_data = [(1, (124, 6)), (2, (62, 3)), (4, (31, 2))]


@pytest.mark.parametrize("batch_size,expected", dl_data)
def test_dataloader(batch_size, expected):
    dvn_dm = bb.dataset.DVNDataModule(batch_size=batch_size, splits=(0.9, 0.05, 0.05))
    dvn_dm.setup()
    train_dl = dvn_dm.train_dataloader()
    val_dl = dvn_dm.val_dataloader()
    test_dl = dvn_dm.test_dataloader()

    assert len(train_dl) == expected[0]
    assert len(test_dl) == len(val_dl) == expected[1]
    if batch_size == 4:
        for i, sample in enumerate(test_dl):
            print(i)
            if i + 1 != len(test_dl):
                assert sample["raw"].shape[0] == 4
            else:
                assert sample["raw"].shape[0] == 2
