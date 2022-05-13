"""Functional interface to several data augmentation functions."""
import torch
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists

from copy import deepcopy
import cv2

class DeepsunCropNonEmptyMaskIfExists(CropNonEmptyMaskIfExists):
    def __init__(self, height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        
        super().__init__(height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0)
        print('Init')
        
    def __call__(self, *args, force_apply=False, **kwargs):
        # print(kwargs)
        mod_kwargs = deepcopy(kwargs)
        # mod_kwargs['image'] = kwargs['sample']['image']
        # mod_kwargs['mask'] = kwargs['segmbyyentation']
        mod_kwargs['mask'] = kwargs['segmentation']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)
#         return processed_kwargs
        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['mask']
        return kwargs

    def apply_to_mask(self, img, **params):        
        X =  super().apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

        return X