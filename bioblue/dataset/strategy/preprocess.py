from typing import List
from bioblue.dataset.numpy import NpzWriter
from pathlib import Path
import zipfile
import SimpleITK as sitk
import cv2
import numpy as np
from . import PrepareStrategy
import logging
import json

log = logging.getLogger(__name__)

interp = dict(image=cv2.INTER_CUBIC, segmentation=cv2.INTER_NEAREST)


class DICOMPrepStrategy(PrepareStrategy):
    def __init__(self, base_dir, directories, resize=False, split=1) -> None:
        self.base_dir = Path(base_dir)
        self.directories = directories
        self.resize = resize
        self.split = split

    def write_files(self, data_dir: Path, latest_files: List[Path]) -> None:
        filenames = []
        assert latest_files is None
        for name, (image_dir, segm_dir) in self.directories.items():
            images_dict = self.get_dicom_images(image=image_dir, segmentation=segm_dir)
            for dtype, images in images_dict.items():
                file_indexes = range(images.GetSize()[2])
                log.debug(f"processing {name} containing {images.GetSize()[2]} slices.")
                split_file_indexes = np.array_split(file_indexes, self.split)
                for file_indexes in split_file_indexes:
                    log.debug(f"{file_indexes}")
                    filename = (
                        data_dir
                        / "train"
                        / dtype
                        / f"{name}_{file_indexes[0]}-{file_indexes[-1]}.npz"
                    )
                    filenames.append(filename.relative_to(data_dir))
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    with NpzWriter(filename) as zf:
                        for idx in file_indexes:
                            log.debug(f"{idx} {images.GetSize()}")
                            image = sitk.GetArrayFromImage(images[:, :, int(idx)])
                            if self.resize:
                                image = cv2.resize(
                                    image, self.resize, interpolation=interp[dtype]
                                )
                            zf.add(image)

        return dict(image=filenames)

    def get_dicom_images(self, **directories):
        images = {}
        for name, directory in directories.items():
            if directory is None:
                continue
            directory = self.base_dir / directory
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
            reader.SetFileNames(dicom_names)
            images[name] = reader.Execute()
        return images


class ImagePrepStrategy(PrepareStrategy):
    def __init__(self) -> None:
        pass
