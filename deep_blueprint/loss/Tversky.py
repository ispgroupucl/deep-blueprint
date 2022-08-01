import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from monai.networks import one_hot
from typing import Callable, List, Optional, Union


ALPHA = 0.5
BETA = 0.5

class TverskyLossV2(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1, alpha=ALPHA, beta=BETA, include_background=True):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, inputs, targets):
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        inputs = F.softmax(inputs,dim=1)       
        
        if not self.include_background:
            inputs = inputs[:,1:]
            targets = targets[:,1:]


        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        return 1 - Tversky

class TverskyLossV3(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1, alpha=ALPHA, beta=BETA, include_background=True, softmax=False, to_onehot_target= False):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background
        self.softmax = softmax
        self.to_onehot_target = to_onehot_target

    def forward(self, input, target):

        # print("shapes: ", input.shape, target.shape)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                # print("softmax")
                input = torch.softmax(input, 1)

        if self.to_onehot_target:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target[:,None,:,:], num_classes=n_pred_ch)


        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # print("Don't take background into account")
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        p0 = input
        p1 = 1 - p0
        g0 = target
        g1 = 1 - g0

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        # if self.batch:
        #     # reducing spatial dimensions and batch
        #     reduce_axis = [0] + reduce_axis

        tp = torch.sum(p0 * g0, reduce_axis)
        fp = self.alpha * torch.sum(p0 * g1, reduce_axis)
        fn = self.beta * torch.sum(p1 * g0, reduce_axis)
        numerator = tp + self.smooth
        denominator = tp + fp + fn + self.smooth

        # print('numerator', numerator , numerator.shape)

        score: torch.Tensor = 1.0 - numerator / denominator
        
        return torch.mean(score)