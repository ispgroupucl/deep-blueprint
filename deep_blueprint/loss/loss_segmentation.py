import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from monai.networks import one_hot
from typing import Callable, List, Optional, Union


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, include_background=True, to_onehot_target= False, softmax = False):
        super().__init__()
        self.include_background = include_background
        self.softmax = softmax
        self.to_onehot_target = to_onehot_target

    def forward(self, inputs, targets, smooth=1):

        # print(inputs.shape, targets.shape)

        n_pred_ch = inputs.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                # print("softmax")
                inputs = torch.softmax(inputs, 1)

        if self.to_onehot_target:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                targets = one_hot(targets[:,None,:,:], num_classes=n_pred_ch)


        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # print("Don't take background into account")
                # if skipping background, removing first channel
                targets = targets[:, 1:]
                inputs = inputs[:, 1:]

        # print(inputs.shape, targets.shape)

        if targets.shape != inputs.shape:
            raise AssertionError(f"ground truth has differing shape ({targets.shape}) from input ({inputs.shape})")

        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
