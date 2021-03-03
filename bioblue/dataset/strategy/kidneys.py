from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import SharedArray as sa
from contextlib import contextmanager
import multiprocessing as mp
from bioblue.dataset.downloadable import md5_dir


@contextmanager
def sharedarray(name, size, dtype=np.float):
    try:
        yield sa.create(f"shm://{name}", size, dtype)
    finally:
        sa.delete(f"shm://{name}")


def readimage(i, scanfile, shape):
    scan = cv2.imread(scanfile, cv2.IMREAD_UNCHANGED)
    scan = cv2.resize(scan, tuple(reversed(shape)), cv2.INTER_CUBIC)
    image = sa.attach("shm://kidneysimage")
    image[:, :, i] = scan


def setup_mip(scanspath=None, image_shape=(512, 512, 1024), mipsize=50):
    if scanspath is None:
        scanspath = Path("~/projects/JPGs - Kidney Ulm - Data Batch 1").expanduser()
    data_dir = Path("~/projects/bio-blueprints/artifacts/data").expanduser()
    dataset_dir = f"kidneys-{'x'.join(str(x) for x in image_shape[:-1])}-mip{mipsize}"

    for i, scandir in enumerate(scanspath.iterdir()):
        print(scandir)
        scanfiles = [x for x in sorted((scandir / "RT/JPG").glob("*.jpg"))]
        if len(scanfiles) - 500 < image_shape[2]:
            image_shape = image_shape[:-1] + (len(scanfiles) - 500,)
        shape = image_shape[:-1]
        z = np.linspace(
            250, len(scanfiles) - 250, image_shape[2], endpoint=False, dtype=np.int
        )
        images = []
        mip_array = np.zeros(shape + (mipsize,), dtype=np.int)
        for i, scanfile_i in enumerate(tqdm(z)):
            scan = cv2.imread(str(scanfiles[scanfile_i]), cv2.IMREAD_UNCHANGED)
            scan = cv2.resize(scan, shape, cv2.INTER_CUBIC)
            mip_array[:, :, i % mipsize] = scan
            images.append(np.max(mip_array, axis=-1))

        (data_dir / dataset_dir / "image").mkdir(exist_ok=True, parents=True)
        np.savez_compressed(
            data_dir / dataset_dir / "image" / f"{scandir.name.replace(' ', '')}.npz",
            *images,
        )

    checksum = md5_dir(data_dir / dataset_dir)
    with open(data_dir / (dataset_dir + ".md5"), "w") as f:
        f.write(checksum + "\n")


def setup(scanspath=None, image_shape=(512, 512, 512), mip=False):
    if scanspath is None:
        scanspath = Path("~/projects/JPGs - Kidney Ulm - Data Batch 1").expanduser()
    images = {}
    data_dir = Path("~/projects/bio-blueprints/artifacts/data").expanduser()
    dataset_dir = f"kidneys-{'x'.join(str(x) for x in image_shape)}"
    with mp.Pool(12) as p, sharedarray("kidneysimage", image_shape) as image:
        for i, scandir in tqdm(enumerate(scanspath.iterdir())):
            print(scandir)
            scanfiles = [str(x) for x in sorted((scandir / "RT/JPG").glob("*.jpg"))][
                250:-250
            ]
            z = np.linspace(
                0, len(scanfiles), image_shape[2], endpoint=False, dtype=np.int
            )
            scanfiles = list(np.array(scanfiles)[z])
            shape = image_shape[:-1]
            args = [(i, x, shape) for i, x in enumerate(scanfiles)]
            p.starmap(readimage, args)
            (data_dir / dataset_dir).mkdir(exist_ok=True)
            np.savez_compressed(
                data_dir / dataset_dir / f"{scandir.name.replace(' ', '')}.npz", image
            )

    checksum = md5_dir(data_dir / dataset_dir)
    with open(data_dir / (dataset_dir + ".md5"), "w") as f:
        f.write(checksum + "\n")


if __name__ == "__main__":
    setup_mip()  # (image_shape=(1024, 1024, 1024))
