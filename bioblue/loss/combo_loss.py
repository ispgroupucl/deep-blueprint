
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from monai.networks import one_hot
from typing import Callable, List, Optional, Union

from hydra.utils import instantiate

COMBO_ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
COMBO_CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

def my_instantiate(infos):
    print(infos)
    lossObject = instantiate(infos)
    return lossObject

class CombineLosses(nn.Module):
    def __init__(self, sublossA, sublossB,
                 ce_ratio=COMBO_CE_RATIO,
                 smooth=1,
                 include_background=True, weight=None) -> None:
        super().__init__()
        
        self.ce_ratio = ce_ratio
        self.smooth = smooth
        self.include_background = include_background

        self.sublossA = sublossA
        self.sublossB = sublossB


    def forward(self, inputs, targets,eps=1e-9):

        loss1 = self.sublossA(inputs, targets)
        loss2 = self.sublossB(inputs, targets)

        combo = (self.ce_ratio * loss1) - ((1 - self.ce_ratio) * loss2)
        
        return combo





class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,
                 alpha=COMBO_ALPHA, ce_ratio=COMBO_CE_RATIO,
                 smooth=1, include_background=True, 
                 softmax=False, to_onehot_target=False):

        super().__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.smooth = smooth
        self.include_background = include_background

        self.softmax = softmax
        self.to_onehot_target = to_onehot_target

    def forward(self, inputs, targets,eps=1e-9):
        print(inputs.shape, targets.shape)
        
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

        if targets.shape != inputs.shape:
            raise AssertionError(f"ground truth has differing shape ({targets.shape}) from input ({inputs.shape})")




        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (self.alpha * ((targets * torch.log(inputs)) + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
        
        return combo