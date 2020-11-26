from bioblue.dataset import synthetic
from bioblue.model import reconstruct
import pytest
from hypothesis import given
import hypothesis.strategies as hs
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
import shutil

from bioblue import __version__
import bioblue as bb
import bioblue


def test_version():
    assert __version__ == "0.1.0"


def test_directorydataset():
    dir_ds = bb.dataset.DirectoryDataset(
        root_dir="./data/Deepvesselnet/labeled", dtypes=["image", "segmentation"]
    )
    assert len(dir_ds) == 136
    first_element = dir_ds[0]
    assert len(first_element) == 2
    assert "image" in first_element and "segmentation" in first_element
    assert len(first_element["image"].shape) == 3
    assert first_element["image"].shape == first_element["segmentation"].shape


dl_data = [(1, (124, 6)), (2, (62, 3)), (4, (31, 2))]


@pytest.mark.parametrize("batch_size,expected", dl_data)
def test_dataloader(batch_size, expected):
    dvn_dm = bb.dataset.DirectoryDataModule(
        directory="Deepvesselnet", batch_size=batch_size, splits=(0.9, 0.05, 0.05)
    )
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
                assert sample["image"].shape[0] == 4
            else:
                assert sample["image"].shape[0] == 2


def test_syntheticdatamodule():
    directory = "synthetic_test"
    train_file = Path("./data") / directory / "train" / "image.npz"
    if (Path("./data") / directory).exists():
        shutil.rmtree(Path("./data") / directory, ignore_errors=True)

    size = 10
    batch_size = 4
    dm = bb.dataset.SyntheticDataModule(
        data_dir="./data",
        directory=directory,
        train_size=size,
        val_size=size,
        test_size=size,
        shape=(512, 512),
        batch_size=batch_size,
    )

    dm.prepare_data()
    assert train_file.exists()
    data = np.load(train_file)
    assert len(data.files) == size
    assert data[data.files[0]].shape == (512, 512)

    dm.setup()
    train_dl = dm.train_dataloader()
    assert len(train_dl) == np.ceil(size / batch_size)


def test_weighted_line():
    img = np.zeros((512, 512))
    yy, xx, val = bb.dataset.weighted_line(0, 0, 511, 511, 1, 0, 512)
    img[xx, yy] = 1

    assert img[256, 256] == 1


def test_lines():
    img = np.zeros((512, 512))
    img = bb.dataset.random_objects(img)
    img, seg = bb.dataset.random_lines(img)
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    assert True


def test_random_erase():
    torch.manual_seed(0)
    img = torch.ones((20, 512, 512))
    erased_img = bb.transforms.random_erase(img)
    assert torch.sum(erased_img == 0) > 0
    assert torch.sum(erased_img == 0) > torch.sum(img == 0)

    torch.manual_seed(0)

    erased_img2 = bb.transforms.random_erase(img, n_blocks=2)
    assert torch.sum(erased_img2 == 0) > torch.sum(erased_img == 0)


@given(
    size=hs.integers(3, 30),
    sigma=hs.floats(0.1, allow_nan=False, allow_infinity=False),
    dimension=hs.integers(2, 3),
)
def test_gaussian_kernel(size, sigma, dimension):
    torch.manual_seed(0)
    # Test sum of kernels equals 1
    kernel = bb.model.gaussian_kernel(size, sigma, dimension)
    assert torch.sum(kernel).allclose(torch.tensor(1.0))
    # assert np.unravel_index(torch.argmax(kernel), kernel.shape) == tuple(
    #     int(x / 2) for x in kernel.shape
    # )


def test_reconstruct():
    torch.manual_seed(0)
    img = torch.ones((512, 512, 5))
    mask = bb.transforms.random_erase(
        img, n_blocks=10, max_size=torch.tensor((64.0, 64.0, 10.0)), return_mask=True
    )
    seg = torch.ones_like(img)
    print(torch.min(mask.to(torch.int)), torch.max(mask.to(torch.int)))
    plt.imshow(mask[:, :, 1:-1].to(torch.float))
    plt.show()
    reconstruct = bioblue.model.ReconstructSegInterpolation()

    img = reconstruct(dict(image=img, mask=mask, segmentation=seg))

    assert img is None


def test_reconstruct_image():
    torch.manual_seed(0)

    sample = bb.dataset.read_sample(
        image="data/HepaticVessel/labeled/image/hepaticvessel_001.nii.gz",
        segmentation="data/HepaticVessel/labeled/segmentation/hepaticvessel_001.nii.gz",
    )

    # assert np.count_nonzero(sample["image"] == 0) == 0

    image, mask = bb.transforms.random_erase(
        torch.tensor(sample["image"]),
        n_blocks=100,
        max_size=torch.tensor((9, 9, 9)),
        return_mask=True,
    )
    sample = {**sample, "mask": mask, "image": image}
    print(sample["image"].shape)
    sample = {name: x[:, :, 30] for name, x in sample.items()}
    sample = {
        name: torch.tensor(x).to(torch.double).unsqueeze_(0).unsqueeze_(0)
        for name, x in sample.items()
    }

    reconstruct = bb.model.ReconstructSegInterpolation()
    img = reconstruct(sample)
    plt.imshow(img[0, 0])
    plt.colorbar()
    plt.show()
    assert False
