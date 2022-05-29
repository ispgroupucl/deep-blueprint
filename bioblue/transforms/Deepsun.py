"""Functional interface to several data augmentation functions."""
import torch

from copy import deepcopy
import cv2
import numpy as np
from skimage.measure import label, regionprops

from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from albumentations.augmentations.geometric.rotate import Rotate, RandomRotate90, SafeRotate
from albumentations.augmentations.transforms import Flip

import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

class DeepsunRandomRotate(SafeRotate):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, 
                        value=None, mask_value=None, always_apply=False, p=0.5):
        super().__init__(limit, interpolation, border_mode, value, mask_value, always_apply, p)

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['masks']
        return kwargs

class DeepsunRandomFlip(Flip):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['masks']
        return kwargs


class DeepsunCropNonEmptyMaskIfExists_SingleMask(CropNonEmptyMaskIfExists):
    def __init__(self, height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        
        super().__init__(height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0)
        # print('Init')
        
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
        # print("CROPNONEMPTY: ", X.shape)
        return X


class DeepsunCropNonEmptyMaskIfExists(CropNonEmptyMaskIfExists):
    def __init__(self, height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        super().__init__(height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0)
        
    def __call__(self, *args, force_apply=False, **kwargs):       
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['masks']
        return kwargs

    def apply_to_mask(self, img, **params):   
        X =  super().apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})
        return X
    
    def apply_to_masks(self, masks, **params):     
        X =  super().apply_to_masks(masks, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})
        return X

    def update_params(self, params, **kwargs):
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = deepcopy(kwargs["masks"])
            mask = self._preprocess_mask(masks[0].astype(np.uint8))
            for m in masks[1:]:
                mask |= self._preprocess_mask(m.astype(np.uint8))
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")
        
        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, self.width - 1)
            y_min = y - random.randint(0, self.height - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height
        # print({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params

class DeepsunMaskMerger(DualTransform):
    def __init__(self, p_add = 0.5,  always_apply=False, p=1.0):     
        super().__init__(always_apply=False, p=1.0)
        self.p_add = p_add
        
    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunMaskMerger')
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        # print(mod_kwargs.keys())
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)
        
        # print("processed", processed_kwargs.keys())
        # print(processed_kwargs['masks'])
        processed_kwargs['mask'] = processed_kwargs['masks']
        del processed_kwargs['masks']

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['mask']
        
        return kwargs

    
    def get_transform_init_args_names(self):
        return ("p_add",)
    
    def apply(self, img, **params):
        return img
    
    
    def apply_to_masks(self, masks, **params):   
        # we consider that the masks are ordered by higher threshold values first
        # -> more small sunspots (maybe fale positives), and large sunspots may merge with close others (LOW)
        # -> less small sunspots (more false negatives), but close sunspots are not merged
        # print('Masks merger')
        
        ####### FIND REGION PROPERTIES IN MASKS
        High_fg_bg = masks[0].copy()
        High_fg_bg[High_fg_bg>0] = 1    
        
        Low_fg_bg = masks[-1].copy()
        Low_fg_bg[Low_fg_bg>0] = 1

        label_low = label(Low_fg_bg)    
        props_labels_low = regionprops(label_low)
        
        
        ####### MERGE THE MASKS
        # 1) Take the High Threshold mask         
        out_mask = masks[0].copy()
        
        # 2) Add the small sunspots in Low threshold mask that do not intersect with other sunspots
        #    with a random condition
        for propLow in props_labels_low:
            bbox= propLow.bbox
            submask = propLow.image 
        
            cur_m_fg_bg = np.zeros_like(out_mask)
            cur_m_fg_bg[bbox[0]:bbox[2], bbox[1]:bbox[3]] = submask
            cur_m = cur_m_fg_bg*masks[-1].copy()
            
            intersection = cur_m * out_mask
            if (np.sum(intersection) == 0)  and  (random.random() < self.p_add):
                out_mask += cur_m
        
        return out_mask