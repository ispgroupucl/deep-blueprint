import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from monai.networks import one_hot
from typing import Callable, List, Optional, Union

FOCAL_ALPHA = 0.8
FOCAL_GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1,
                    alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                    softmax=False, to_onehot_target= False, 
                    include_background=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.smooth = smooth
        self.softmax = softmax
        self.to_onehot_target = to_onehot_target
        self.include_background = include_background

    def forward(self, inputs, targets,):
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

        # #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss


FOCAL_TVERSKY_ALPHA = 0.5
FOCAL_TVERSKY_BETA = 0.5
FOCAL_TVERSKY_GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1,
                    alpha=FOCAL_TVERSKY_ALPHA, beta=FOCAL_TVERSKY_BETA, gamma=FOCAL_TVERSKY_GAMMA,
                    softmax=False, to_onehot_target= False, 
                    include_background=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.smooth = smooth
        self.softmax = softmax
        self.to_onehot_target = to_onehot_target
        self.include_background = include_background

    def forward(self, inputs, targets):
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

        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky
