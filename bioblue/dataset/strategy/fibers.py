from numba import njit
from skimage.filters.thresholding import threshold_multiotsu
from bioblue.dataset.numpy import NpzWriter
from pathlib import Path
from skimage import transform
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
from . import Strategy
from bioblue import fibers
import logging

log = logging.getLogger(__name__)


class FiberCropStrategy(Strategy):
    def __init__(
        self, input_dtype="image", partition="train", crop_size=256, grid_size=200,
    ) -> None:
        self.input_dtype = input_dtype
        self.partition = partition
        self.crop_size = crop_size
        self.grid_size = grid_size

    def index_generator(self, image):
        xx, yy = np.mgrid[
            self.crop_size // 2 : image.shape[1]
            - self.crop_size // 2 : self.grid_size // 2,
            self.crop_size // 2 : image.shape[0]
            - self.crop_size // 2 : self.grid_size // 2,
        ]
        xx, yy = xx.flatten(), yy.flatten()
        return zip(xx, yy)

    def prepare_crop(self, index, image):
        x, y = index
        idx_name = f"{x}_{y}"
        crop = image[
            y - self.crop_size // 2 : y + self.crop_size // 2,
            x - self.crop_size // 2 : x + self.crop_size // 2,
        ]
        if crop.min() < self.thresh:  # TODO : don't use constant here
            return idx_name, None

        angle, confidence = orientation(crop)
        if confidence < 0.9:
            return idx_name, None
        rotated_crop = rotate_and_crop(image, (x, y), 90 - angle, self.crop_size)
        if rotated_crop.min() < 55:
            return idx_name, None

        return idx_name, rotated_crop

    # def prepare_image(self, image):
    #     self.thresh = threshold_multiotsu(image, classes=4)[0]
    #     for index in self.index_generator(image):
    #         idx_name, crop = self.prepare_crop(index, image)
    #         if crop is not None:
    #             filename = (
    #                 self.data_dir
    #                 / self.partition
    #                 / self.input_dtype
    #                 / f"{self.np_file.stem}_{idx_name}.npz"
    #             )
    #             zf = self.zf_dict.set_default(idx_name, NpzWriter(filename))
    #             zf.add(crop)

    def prepare_data(self, data_dir: Path) -> None:
        if (data_dir / self.partition / self.input_dtype).exists():
            return  # TODO : always exits...
        for np_file in (data_dir / self.partition / self.input_dtype).iterdir():
            log.debug(f"processing {np_file.name}")
            sample = np.load(np_file)
            zf_dict = {}
            for slice_name in sample:
                log.debug(f"processing slice {slice_name}")
                slice = sample[slice_name]
                self.thresh = threshold_multiotsu(slice, classes=4)[0]
                for index in self.index_generator(slice):
                    idx_name, crop = self.prepare_crop(index, slice)
                    if crop is not None:
                        filename = (
                            data_dir
                            / self.partition
                            / self.input_dtype
                            / f"{np_file.stem}_{idx_name}.npz"
                        )
                        if idx_name not in zf_dict:
                            log.debug(f"creating new writer for {idx_name}")
                            zf_dict[idx_name] = NpzWriter(filename)
                        zf = zf_dict[idx_name]
                        zf.add(crop)
                # Close all NpzWriters
            for zf in zf_dict.values():
                zf.close()


class FiberSegStrategy(Strategy):
    def __init__(
        self, input_dtype="image", partition="train", dtype="segmentation"
    ) -> None:
        self.partition = partition
        self.input_dtype = input_dtype
        self.dtype = dtype

    def prepare_data(self, data_dir: Path) -> None:
        for np_file in (data_dir / self.partition / self.input_dtype).iterdir():
            log.debug(f"processing {np_file.name}")
            sample = np.load(np_file)
            filename = data_dir / self.partition / self.dtype / np_file.name
            filename.parent.mkdir(parents=True, exist_ok=True)
            with NpzWriter(filename) as zf:
                for slicename in sample:
                    slice = sample[slicename]
                    segmentation = fibers.find_1d_peaks(slice, segment_valleys=True)
                    zf.add(segmentation)


def orientation(crop, num=100):
    ft = np.fft.fft2(crop)
    # Remove constant term
    ft[0, :] = 1
    ft[:, 0] = 1
    ft = np.fft.fftshift(ft)
    ft_logabs = np.log(np.abs(ft))
    # data = np.nonzero(ft_logabs > minimum)
    data = np.unravel_index(np.argsort(-ft_logabs, axis=None)[:num], shape=crop.shape)
    ft_filtered = np.zeros_like(crop)
    ft_filtered[data[0], data[1]] = 1
    # ft_filtered = ft_logabs > minimum
    if len(data[0]) == 0:
        raise NotImplementedError()
    pca = make_pipeline(
        StandardScaler(with_mean=True, with_std=False), PCA(n_components=2)
    )
    pca = pca.fit(np.vstack(list(reversed(data))).T)
    pca: PCA = pca.named_steps["pca"]
    angle = compute_angle(pca.components_[0])
    confidence = pca.explained_variance_ratio_[0]  # TODO: compute
    return angle, confidence


@njit(cache=True, nogil=True)
def compute_angle(vector):
    """ Returns the angle in radians between given vectors"""
    v1_u = vector / np.linalg.norm(vector)
    v2_u = np.array([1.0, 0.0])
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    if minor == 0:
        sign = 1
    else:
        sign = np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return np.degrees(sign * np.arccos(dot_p))


def rotate_and_crop(image, center, angle, size):
    image = transform.rotate(image.copy(), angle, center=center) * 255
    return image[
        center[1] - size // 2 : center[1] + size // 2,
        center[0] - size // 2 : center[0] + size // 2,
    ]
