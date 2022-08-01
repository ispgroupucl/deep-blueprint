from typing import Dict, List
from ..numpy import NpzWriter
from pathlib import Path
import zipfile
import cv2
import numpy as np
from . import PrepareStrategy
import logging
import json

log = logging.getLogger(__name__)

interp = dict(image=cv2.INTER_CUBIC, segmentation=cv2.INTER_NEAREST)


class SummaryStrategy(PrepareStrategy):
    def write_files(
        self, data_dir: Path, latest_files: Dict[str, Dict[str, List[Path]]]
    ) -> Dict[str, Dict[str, List[Path]]]:
        summary = {}
        assert latest_files is None
        for split_dir in data_dir.iterdir():
            if not split_dir.is_dir():
                continue
            summary[split_dir.name] = {}
            for cat_dir in split_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                summary[split_dir.name][cat_dir.name] = sorted(cat_dir.iterdir())
        return summary


class ImagePrepStrategy(PrepareStrategy):
    def __init__(self) -> None:
        pass
