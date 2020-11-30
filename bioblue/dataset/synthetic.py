from bioblue.dataset.utils import NumpyDataset
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import draw
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import logging
from filelock import FileLock
import json
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl


log = logging.getLogger(__name__)


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        shape=(512, 512),
        train_size=50,
        val_size=50,
        test_size=50,
        data_dir: str = "./data",
        directory: str = "synthetic",
        batch_size=2,
        num_workers=1,
        points=200,
        links=2,
    ):
        super().__init__()
        self.sizes = dict(train=train_size, val=val_size, test=test_size)
        self.data_dir = Path(data_dir)
        self.dirprefix = directory
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.arguments = self.get_init_arguments_and_types()
        self.points = points
        self.links = 2
        self.lock = FileLock(self.data_dir / (self.dirname + ".lock"))
        self.dir = Path(data_dir) / self.dirname

    @property
    def dirname(self):
        return f"{self.dirprefix}_s{self.sizes['train']}-{self.sizes['val']}-{self.sizes['test']}_p{self.points}_l{self.links}"

    def prepare_data(self) -> None:
        with self.lock.acquire():
            if self.dir.exists():
                return
            self.dir.mkdir(parents=True, exist_ok=True)
            for name, size in self.sizes.items():
                log.info(f"Creating {name}")
                imgs = []
                segms = []
                shapes = size * [self.shape]
                with mp.Pool(12) as pool:
                    results = pool.map(create_random_image, shapes)
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
    img = np.zeros(shape)
    rng = np.random.default_rng()
    points = rng.integers(100, 500)
    links = rng.integers(3, 20)
    background = random_objects(img)
    cimg, segm = random_lines(background, points, links)

    return cimg, segm


def random_objects(
    img, max_shapes=1000, min_size=50, max_size=500, intensity_range=(0, 50)
):
    drawing, mask = draw.random_shapes(
        img.shape,
        max_shapes=max_shapes,
        min_size=min_size,
        max_size=max_size,
        multichannel=False,
        intensity_range=intensity_range,
        allow_overlap=True,
        num_trials=1000,
    )
    drawing[drawing == 255] = 0
    return drawing


def random_lines(img, points=200, links=2):
    rng = np.random.default_rng()
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
            *line_coords[coord], w=rng.integers(5), rmax=img.shape[0]
        )
        img[xx, yy] = rng.integers(256)
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

