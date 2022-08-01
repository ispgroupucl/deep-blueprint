import logging
from typing import Dict, Optional, Union
from albumentations import DualTransform
from matplotlib import pyplot as plt
from monai.transforms import MapTransform, RandomizableTransform, InvertibleTransform
import numpy as np
import random

log = logging.getLogger(__name__)


class BinarySegmentation(DualTransform):
    def __init__(self, keep: 1):
        self.keep = keep
        super().__init__(always_apply=True)

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **kwargs):
        mask[mask != self.keep] = 0
        mask[mask == self.keep] = 1
        return mask


class RandomBackgroundAdd(DualTransform):
    def __init__(self, always_apply=False, p=1, p_background=0.1):
        super().__init__(always_apply=always_apply, p=p)
        self.p_background = p_background

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, include_mask, **params):
        mask[include_mask == True] = 2
        return mask

    def get_params_dependent_on_targets(self, params):
        mask = params["segmentation"]
        r_mask = np.random.random_sample(mask.shape)
        include_mask = (mask == 0) & (r_mask < self.p_background)
        params.update({"include_mask": include_mask})
        return params

    @property
    def targets_as_params(self):
        return ["segmentation"]

    def get_transform_init_args_names(self):
        return ("p_background", "p_valley")


class DropSegmentation(DualTransform):
    def __init__(self, always_apply=False, p=1.0, p_dropout=0.5):
        self.p_dropout = p_dropout
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, remove_mask, **params):
        mask[remove_mask == True] = 0
        return mask

    def get_params_dependent_on_targets(self, params):
        mask = params["segmentation"]
        r_mask = np.random.random_sample(mask.shape)
        remove_mask = (mask == 1) & (r_mask < self.p_dropout)
        params.update(dict(remove_mask=remove_mask))
        return params

    @property
    def targets_as_params(self):
        return ["segmentation"]

    def get_transform_init_args_names(self):
        return super().get_transform_init_args_names(["p_dropout"])


class MapLabels(MapTransform):
    def __init__(
        self, keys=("segmentation",), mapping: Optional[Dict[int, int]] = None
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.mapping = mapping

    def __call__(self, data):
        if self.mapping is None:
            return data
        for key in self.key_iterator(data):
            assert key == "segmentation"
            for inmap, outmap in self.mapping.items():
                data[key][data[key] == inmap] = outmap
        return data


class RandomBackgroundAddMonai(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys=("segmentation",),
        prob: float = 1,
        p_background: float = 0.1,
        allow_missing_keys=False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.p_background = p_background
        self.include_mask = None

    def randomize(self, data) -> None:
        super().randomize(None)
        mask = data["segmentation"]
        r_mask = self.R.random_sample(mask.shape)
        self.include_mask = (mask == 0) & (r_mask < self.p_background)

    def __call__(self, data, randomize=True):
        if randomize:
            self.randomize(data)
        for key in self.key_iterator(data):
            data[key][self.include_mask == True] = 2
        return data


class RandomSegmThreshold(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys=("segmentation",),
        base_key="image",
        prob: float = 1,
        scale: float = 2,
        thresh_neg: Optional[int] = None,
        thresh_pos: Optional[int] = None,
        allow_missing_keys=False,
    ):
        RandomizableTransform.__init__(self, prob=prob)
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.scale = scale
        self.thresh_pos = thresh_pos
        self.thresh_neg = thresh_neg
        self.rand_mask = None
        self.base_key = base_key

    def randomize(self, data) -> None:
        super().randomize(None)
        mask = data[self.base_key]
        r_mask = self.R.normal(mask, self.scale)
        self.rand_mask = np.clip(r_mask, 0, 255)
        # fig, axs = plt.subplots(ncols=3, figsize=(30, 10))
        # axs[0].imshow(mask[0, :, mask.shape[1] // 2], cmap="gray")
        # axs[1].imshow(self.rand_mask[0, :, self.rand_mask.shape[1] // 2], cmap="gray")
        # diff_plot = axs[2].imshow(
        #     self.rand_mask[0, :, self.rand_mask.shape[1] // 2]
        #     - mask[0, :, mask.shape[1] // 2]
        # )
        # plt.colorbar(diff_plot, ax=axs[2])
        # plt.show()

    def __call__(self, data, randomize=True):
        if randomize:
            self.randomize(data)
        if self.thresh_neg is None:
            thresh_neg = data[self.base_key].min()
        else:
            thresh_neg = self.thresh_neg
        thresh_pos = (
            self.thresh_pos
            if self.thresh_pos is not None
            else data[self.base_key].max()
        )
        for key in self.key_iterator(data):
            mask = data[key]
            to_neg = (mask == 0) & (self.rand_mask <= thresh_neg)
            to_pos = (mask == 0) & (self.rand_mask >= thresh_pos)
            log.debug(
                "neg:",
                np.count_nonzero(to_neg),
                np.count_nonzero((self.rand_mask <= thresh_neg)),
            )
            log.debug(
                "pos:",
                np.count_nonzero(to_pos),
                np.count_nonzero(self.rand_mask >= thresh_pos),
            )
            data[key][to_neg] = 2
            data[key][to_pos] = 1
            data[f"_{key}_neg"] = np.count_nonzero(to_neg)
            data[f"_{key}_pos"] = np.count_nonzero(to_pos)
        return data


class DropSegmentationMonai(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys=("segmentation",),
        prob: float = 1,
        p_dropout=0.5,
        allow_missing_keys=False,
    ):
        RandomizableTransform.__init__(self, prob=prob)
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.p_dropout = p_dropout
        self.remove_mask = None

    def randomize(self, data) -> None:
        super().randomize(None)
        mask = data[self.keys[0]]
        r_mask = self.R.random_sample(mask.shape)
        self.remove_mask = (mask == 1) & (r_mask < self.p_dropout)

    def __call__(self, data, randomize=True):
        if randomize:
            self.randomize(data)
        for key in self.key_iterator(data):
            data[key][self.remove_mask == True] = 0
        return data
