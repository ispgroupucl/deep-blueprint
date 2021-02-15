from typing import Dict, Mapping, Tuple
from bioblue.dataset.utils import NumpyDataset
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.filters import gaussian_filter
from skimage import draw
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import logging
from filelock import FileLock
import json
import yaml
import hashlib
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args
import inspect
from omegaconf import Container
from hydra.utils import call
from functools import partial
from hydra._internal.utils import _locate

log = logging.getLogger(__name__)


class SyntheticDataModule(pl.LightningDataModule):
    """ Module for creating a synthetic datamodule.
    
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (512, 512),
        train_size: int = 50,
        val_size: int = 50,
        test_size: int = 50,
        data_dir: str = "./data",
        directory: str = "synthetic",
        batch_size: int = 2,
        num_workers: int = 1,
        image_creator: Mapping = None,
        points_range=(20, 21),  # DEPRECATED
        links_range=(3, 4),  # DEPRECATED
        max_shapes=1000,  # DEPRECATED
        size_range=(50, 500),  # DEPRECATED
        weight_range=(1, 5),  # DEPRECATED
        bg_intensity_range=(0, 50),  # DEPRECATED
        fg_intensity_range=(0, 256),  # DEPRECATED
    ):
        super().__init__()
        self.sizes = dict(train=train_size, val=val_size, test=test_size)
        self.data_dir = Path(data_dir)
        self.dirprefix = directory
        self.shape = tuple(shape)
        self.batch_size = batch_size
        self.num_workers = num_workers
        if image_creator is not None:
            target = _locate(image_creator.pop("_target_"))
            image_creator["shape"] = tuple(shape)
            self.image_creator = partial(target, **image_creator)
        else:
            parameters = (
                tuple(shape),
                tuple(points_range),
                tuple(links_range),
                max_shapes,
                tuple(size_range),
                tuple(weight_range),
                tuple(bg_intensity_range),
                tuple(fg_intensity_range),
            )
            self.image_creator = partial(create_image, *parameters)
            log.warn(
                "You should pass the function you want to call explicitly using image_creator"
            )
        # TODO : find solution to this mess
        self.arguments = get_init_args(inspect.currentframe())
        for name, argument in self.arguments.items():
            self.arguments[name] = (
                tuple(argument) if isinstance(argument, Container) else argument
            )
        self.lock = FileLock(self.data_dir / (self.dirname + ".lock"))
        self.dir = Path(data_dir) / self.dirname

    @property
    def dirname(self):
        h = hashlib.sha1(yaml.safe_dump(self.arguments).encode()).hexdigest()
        return f"{self.dirprefix}_{h[:10]}"

    def prepare_data(self) -> None:
        with self.lock.acquire():
            if self.dir.exists():
                return
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "config.yaml", "w") as f:
                yaml.safe_dump(self.arguments, f)
            for name, size in self.sizes.items():
                log.info(f"Creating {name}")
                imgs = []
                segms = []
                parameters = size * [()]
                with mp.Pool(12) as pool:
                    results = pool.starmap(self.image_creator, parameters)
                imgs = [x[0] for x in results]
                segms = [x[1] for x in results]
                (self.dir / name).mkdir()
                for dtype, arrays in zip(["image", "segmentation"], [imgs, segms]):
                    np.savez_compressed(self.dir / name / (dtype + ".npz"), *arrays)

    def setup(self, stage=None):
        self.train = NumpyDataset(self.dir / "train", ["image", "segmentation"])
        self.val = NumpyDataset(self.dir / "val", ["image", "segmentation"])
        self.test = NumpyDataset(self.dir / "test", ["image", "segmentation"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


def create_random_image(shape):
    rng = np.random.default_rng()
    points = rng.integers(10, 500)
    links = rng.integers(3, 20)
    cimg, segm = create_image(shape, points, links)
    return cimg, segm


def create_shapes(
    shape: Tuple[int, int],
    foreground_params: Mapping = None,
    background_params: Mapping = None,
    uniform_background=True,
    blur_sigma=10,
):
    rng = np.random.default_rng()
    img = np.zeros(shape)
    if not uniform_background:
        img, _ = random_objects(img, **background_params)
        img = gaussian_filter(img, sigma=blur_sigma)
    else:
        img[:] = rng.integers(0, 128)

    img, segm = random_objects(img, **foreground_params)

    return img, segm


def create_lines(
    shape: Tuple[int, int],
    foreground_params: Mapping = None,
    background_params: Mapping = None,
    uniform_background=True,
    blur_sigma=10,
) -> Tuple[np.array, np.array]:
    rng = np.random.default_rng()
    img = np.zeros(shape)
    if not uniform_background:
        img, _ = random_objects(img, **background_params)
        img = gaussian_filter(img, sigma=blur_sigma)
    else:
        img[:] = rng.integers(0, 128)

    img, segm = random_lines(img, **foreground_params)

    return img, segm


def create_image(
    shape,
    points_range=(20, 21),
    links_range=(3, 4),
    max_shapes=1000,
    size_range=(50, 500),
    weight_range=(1, 5),
    bg_intensity_range=(0, 50),
    fg_intensity_range=(0, 256),
):
    rng = np.random.default_rng()
    points = rng.integers(*points_range)
    links = rng.integers(*links_range)
    img = np.zeros(shape)
    background, _ = random_objects(img, max_shapes, *size_range, bg_intensity_range)
    cimg, segm = random_lines(
        background, points, links, fg_intensity_range, weight_range
    )

    return cimg, segm


def random_objects(
    img,
    max_shapes=1000,
    min_size=50,
    max_size=500,
    intensity_range=(0, 50),
    min_shapes=1,
    allow_overlap=True,
):
    drawing, _ = draw.random_shapes(
        img.shape,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        min_size=min_size,
        max_size=max_size,
        multichannel=False,
        intensity_range=intensity_range,
        allow_overlap=allow_overlap,
        num_trials=1000,
    )
    # drawing[drawing == 255] = 0
    img[drawing != 255] = drawing[drawing != 255]
    mask = np.zeros_like(img)
    mask[drawing != 255] = 1
    return img, mask


def random_lines(
    img,
    points=200,
    links=2,
    intensity_range=(0, 256),
    weight_range=(1, 5),
    points_range=None,
    links_range=None,
):
    rng = np.random.default_rng()
    if points_range is not None:
        points = rng.integers(*points_range)
    if links_range is not None:
        links = rng.integers(*links_range)
    links = min(points - 1, links)  # In case there are too few points
    p = rng.integers(low=(0, 0), high=img.shape, size=(points, 2))
    p = np.unique(p, axis=0)  # remove same points
    points = len(p)
    dist = squareform(pdist(p))
    dist_idx = np.argsort(dist, axis=-1)
    dist_links = np.stack(
        [np.stack(links * [np.arange(points)], axis=-1), dist_idx[:, 1 : links + 1]],
        axis=-1,
    )
    dist_links = np.unique(np.sort(dist_links, axis=-1).reshape(-1, 2), axis=0)
    line_coords = p[dist_links].reshape(-1, 4)
    segm = np.zeros_like(img)
    for coord in range(len(line_coords)):
        xx, yy, val = weighted_line(
            *line_coords[coord], w=rng.integers(*weight_range), rmax=img.shape[0]
        )
        img[xx, yy] = rng.integers(*intensity_range)
        segm[xx, yy] = 1
    return img, segm


def trapez(y, y0, w):
    return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    if abs(c1 - c0) < abs(r1 - r0):
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    slope = (r1 - r0) / (c1 - c0)
    w *= np.sqrt(1 + np.abs(slope)) / 2

    x = np.arange(c0, c1 + 1, dtype=float)
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)

    thickness = np.ceil(w / 2)
    yy = np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(
        1, -1
    )
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1, 1), w).flatten()

    yy = yy.flatten()

    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

