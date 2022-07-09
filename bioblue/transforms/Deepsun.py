"""Functional interface to several data augmentation functions."""
from configparser import Interpolation
from tokenize import group
from kornia import center_crop
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

from scipy.ndimage.interpolation import rotate

import matplotlib.pyplot as plt

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

class DeepsunRandomMaskSelector(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(always_apply=False, p=1.0)
        
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
    
    def apply(self, img, **params):
        return img
    
    
    def apply_to_masks(self, masks, **params):  
        rnd_idx =  random.randint(0,len(masks)-1)
        # print(rnd_idx) 
        out_mask = masks[rnd_idx].copy()
       
        return out_mask

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


def rotate_CV_bound(image, angle, interpolation):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),flags=interpolation)
class DeepsunRotateAndCropAroundGroup(DualTransform):

    def __init__(self, standard_height, standard_width,  always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.standard_height = standard_height
        self.standard_width = standard_width

        self.index = 0

    def padder(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')
        img = kwargs['image']
        msk = kwargs['mask']

        # 1) Correct solar Angle ->  Rotate image + Zoom In
        angle = kwargs['solar_angle']
        deltashapeX = kwargs['deltashapeX']
        deltashapeY = kwargs['deltashapeY']
        # print('solar_angle', angle)

        # rot_img = rotate(img, angle=angle, reshape=True)
        # rot_msk = rotate(msk, angle=angle, reshape=True)
        rot_img = rotate_CV_bound(img, angle=angle, interpolation=cv2.INTER_LINEAR)
        rot_msk = rotate_CV_bound(msk, angle=angle, interpolation=cv2.INTER_LINEAR)

        rot_img_zoom = rot_img[deltashapeX//2:rot_img.shape[0]-deltashapeX//2,
                          deltashapeY//2:rot_img.shape[1]-deltashapeY//2] 
        rot_msk_zoom = rot_msk[deltashapeX//2:rot_msk.shape[0]-deltashapeX//2,
                          deltashapeY//2:rot_msk.shape[1]-deltashapeY//2] 

        # print(rot_img_zoom.shape, rot_msk_zoom.shape)
        assert rot_img_zoom.shape == rot_msk_zoom.shape

        # 2) Crop around group
        group_centroid = np.array(kwargs['centroid_px'])
        # print(group_centroid)

        # minX = int(group_centroid[0])-self.standard_width//2
        # maxX = int(group_centroid[0])+self.standard_width//2
        # minY = int(group_centroid[1])-self.standard_height//2
        # maxY = int(group_centroid[1])+self.standard_height//2

        # img_group_crop = rot_img_zoom[minX:maxX,minY:maxY]
        # msk_group_crop = rot_msk_zoom[minX:maxX,minY:maxY]

        minX = self.standard_height + (int(group_centroid[1])-self.standard_width//2)
        maxX = self.standard_height + (int(group_centroid[1])+self.standard_width//2)
        minY = self.standard_height + (int(group_centroid[0])-self.standard_height//2)
        maxY = self.standard_height + (int(group_centroid[0])+self.standard_height//2)


        pad_rot_img_zoom = np.pad(rot_img_zoom, self.standard_height, self.padder, padder=0)
        pad_rot_msk_zoom = np.pad(rot_msk_zoom, self.standard_height, self.padder, padder=0)

        img_group_crop = pad_rot_img_zoom[minX:maxX,minY:maxY]
        msk_group_crop = pad_rot_msk_zoom[minX:maxX,minY:maxY]

        assert img_group_crop.shape == msk_group_crop.shape

        # print(msk_group_crop.shape)

        kwargs.pop('solar_angle',None)
        kwargs.pop('deltashapeX',None)
        kwargs.pop('deltashapeY',None)
        kwargs.pop('centroid_px',None)

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        # ax[0].imshow(img, interpolation=None, cmap='gray')
        # ax[0].imshow(msk, interpolation=None, alpha=0.5)
        # ax[1].imshow(rot_img_zoom, interpolation=None, cmap='gray')
        # ax[1].imshow(rot_msk_zoom, interpolation=None, alpha=0.5)
        # ax[2].imshow(img_group_crop, interpolation=None, cmap='gray')
        # ax[2].imshow(msk_group_crop, interpolation=None, alpha=0.5)
        # ax[1].scatter(group_centroid[0],group_centroid[1], c='r', s=1 )
        # plt.savefig(f'./test_classification_{self.index}.png', dpi=150)
        
        self.index+=1


        kwargs['image'] = img_group_crop.copy()
        kwargs['mask'] = msk_group_crop.copy()
        
        return kwargs

class DeepsunImageMaskProduct(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')
        img = kwargs['image'].copy()
        msk = kwargs['mask'].copy()

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        # ax[0].imshow(img, interpolation=None, cmap='gray')
        # ax[1].imshow(msk, interpolation=None)
        # ax[2].imshow(img * msk, interpolation=None, cmap='gray')
        # plt.savefig(f'./test_classification_MaskIm{self.index}.png', dpi=150)

        self.index+=1

        kwargs['image'] = img * msk

        return kwargs



    