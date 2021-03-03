from contextlib import contextmanager
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
import SharedArray as sa
import uuid
import multiprocessing as mp
import click
from typing import Tuple, Optional
import typer


@contextmanager
def sharedarray(name, size, dtype=np.float):
    try:
        yield sa.create(f"shm://{name}", size, dtype)
    finally:
        sa.delete(f"shm://{name}")


def readimage(i, scanfile, shape, name):
    scan = cv2.imread(scanfile, cv2.IMREAD_UNCHANGED)
    scan = cv2.resize(scan, tuple(reversed(shape)), cv2.INTER_CUBIC)
    image = sa.attach(f"shm://{name}")
    image[:, :, i] = scan


class DefaultSetup:
    def __init__(
        self,
        indir: Path,
        outdir: Path,
        image_shape: Tuple[int, int] = (None, None),
        slicenumber: Optional[int] = None,
        slicerange: Tuple[int, int] = (0, -1),
        mipsize: Optional[int] = None,
    ):
        print(type(indir))
        self.indir = Path(indir)
        self.outdir = Path(outdir)
        self.image_shape = image_shape
        self.slicenumber = slicenumber
        self.slicerange = slicerange
        self.mipsize = mipsize
        self.name = str(uuid.uuid4())
        self.setup()

    def setup(self):
        with mp.Pool() as p:
            for i, scandir in enumerate(self.indir.iterdir()):
                print(scandir.name)
                tmppath = next(scandir.glob("**/*.*"))
                suffix = tmppath.suffix
                prefix = tmppath.relative_to(scandir).parent
                scanfiles = [
                    str(x) for x in sorted((scandir / prefix).glob(f"*{suffix}"))
                ][self.slicerange[0] : self.slicerange[1]]
                slicenumber = (
                    self.slicenumber if self.slicenumber is not None else len(scanfiles)
                )
                z = np.linspace(
                    0, len(scanfiles), slicenumber, endpoint=False, dtype=np.int
                )
                scanfiles = list(np.array(scanfiles)[z])
                if self.image_shape[0] is None:
                    shape = cv2.imread(scanfiles[0], cv2.IMREAD_UNCHANGED).shape
                else:
                    shape = self.image_shape

                full_shape = self.image_shape + (slicenumber,)
                with sharedarray(self.name, full_shape) as image:
                    args = [(i, x, shape, self.name) for i, x in enumerate(scanfiles)]
                    p.starmap(readimage, args)
                    (self.outdir / "image").mkdir(exist_ok=True, parents=True)
                    np.savez_compressed(
                        self.outdir / "image" / f"{scandir.name.replace(' ', '')}.npz",
                        *image.transpose(2, 0, 1),
                    )
                    if self.mipsize is not None:
                        (self.outdir / "mip").mkdir(exist_ok=True, parents=True)
                        images = []
                        for slice_i in range(image.shape[2]):
                            images.append(
                                np.max(
                                    image[:, :, slice_i : slice_i + self.mipsize],
                                    axis=-1,
                                )
                            )
                        np.savez_compressed(
                            self.outdir
                            / "mip"
                            / f"{scandir.name.replace(' ', '')}.npz",
                            *images,
                        )


# def main():
#     setup = DefaultSetup(args.indir, args.outdir)
#     setup.setup()

if __name__ == "__main__":
    setup = typer.run(DefaultSetup)
