from argparse import ArgumentParser
import os

from filelock import FileLock

from bioblue.dataset.utils import NumpyDataset
from pathlib import Path
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import hashlib
from minio import Minio
from minio.error import NoSuchKey

log = logging.getLogger(__name__)


class DownloadableDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        directory: str = "kidneys",
        batch_size=2,
        num_workers=1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.directory = directory
        self.dir = Path(data_dir) / directory
        self.lock = FileLock(self.data_dir / ("." + self.directory + ".lock"))

    def prepare_data(self) -> None:
        with self.lock.acquire():
            mclient = Minio(
                os.environ["MLFLOW_S3_ENDPOINT_URL"].replace("http://", ""),
                access_key=os.environ["AWS_ACCESS_KEY_ID"],
                secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                secure=False,
            )
            if self.dir.exists():
                dirhash = md5_dir(self.dir)
                server_dirhash = ""
                try:
                    response = mclient.get_object("data", self.directory + ".md5")
                except NoSuchKey:
                    log.warn("There is no hashfile on the server.")
                    return
                try:
                    server_dirhash = response.read(decode_content=True).strip()
                finally:
                    response.close()
                    response.release_conn()
                print(dirhash, server_dirhash)
                if dirhash == server_dirhash:
                    return
                else:
                    raise FileExistsError(
                        "current directory is not the same as the one on the server."
                    )

            objects = mclient.list_objects_v2(
                "data", prefix=self.directory + "/", recursive=True
            )
            for obj in objects:
                local_path = self.dir / obj.object_name.split("/", 1)[1]
                local_path.parent.mkdir(parents=True, exist_ok=True)
                mclient.fget_object("data", obj.object_name, str(local_path))

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


def md5_update_from_dir(directory, hash):
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash.update(chunk)
        elif path.is_dir():
            hash = md5_update_from_dir(path, hash)
    return hash


def md5_dir(directory):
    return md5_update_from_dir(directory, hashlib.md5()).hexdigest()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    checksum = md5_dir(args.directory)
    with open(args.directory + ".md5", "w") as f:
        f.write(checksum + "\n")

