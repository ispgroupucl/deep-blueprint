import os

from bioblue.dataset.utils import NumpyDataset
from pathlib import Path
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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

    def prepare_data(self) -> None:
        if not self.dir.exists():
            from minio import Minio

            mclient = Minio(
                os.environ["MLFLOW_S3_ENDPOINT_URL"].replace("http://", ""),
                access_key=os.environ["AWS_ACCESS_KEY_ID"],
                secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                secure=False,
            )
            objects = mclient.list_objects_v2(
                "data", prefix=self.directory, recursive=True
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
