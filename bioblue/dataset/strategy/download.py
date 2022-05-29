import os

from filelock import FileLock

from pathlib import Path
import logging
import hashlib
from minio import Minio
from minio.error import NoSuchKey

from . import PrepareStrategy

log = logging.getLogger(__name__)


class DownloadStrategy(PrepareStrategy):
    def __init__(self) -> None:
        pass

    def prepare_data(self, data_dir: Path) -> None:
        self.lock = FileLock(data_dir.parent / ("." + data_dir.name + ".lock"))
        with self.lock.acquire():
            mclient = Minio(
                os.environ["MLFLOW_S3_ENDPOINT_URL"].replace("http://", ""),
                access_key=os.environ["AWS_ACCESS_KEY_ID"],
                secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                secure=False,
            )
            if data_dir.exist():
                dirhash = md5_dir(data_dir)
                server_dirhash = ""
                try:
                    response = mclient.get_object("data", data_dir.name + ".md5")
                except NoSuchKey:
                    log.error("There is no hashfile on the server")
                    return
                try:
                    server_dirhash = response.read(decode_content=True).decode().strip()
                finally:
                    response.close()
                    response.release_conn()
                if dirhash == server_dirhash:
                    return
                else:
                    raise FileExistsError("syncing error with server.")

            objects = mclient.list_objects_v2(
                "data", prefix=data_dir.name + "/", recursive=True
            )
            for obj in objects:
                local_path = data_dir / obj.object_name.split("/", 1)[1]
                local_path.parent.mkdir(parents=True, exist_ok=True)
                mclient.fget_object("data", obj.object_name, str(local_path))


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
