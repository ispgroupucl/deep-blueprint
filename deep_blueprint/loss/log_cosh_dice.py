from distutils.ccompiler import gen_lib_options
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from monai.networks import one_hot
from typing import Callable, List, Optional, Union


class LogCosHDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth= 1., include_background=True,
                 softmax=False, to_onehot_target= False) -> None:
        super().__init__()
        self.smooth = smooth
        self.softmax = softmax
        self.to_onehot_target = to_onehot_target
        self.include_background = include_background

    def forward(self, inputs, targets):
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

        if targets.shape != inputs.shape:
            raise AssertionError(f"ground truth has differing shape ({targets.shape}) from input ({inputs.shape})")


        return self.log_cosh_dice_loss(targets, inputs)

    def generalized_dice_coefficient(self, y_true, y_pred):
            y_true_f = torch.flatten(y_true)
            y_pred_f = torch.flatten(y_pred)
            intersection = torch.sum(y_true_f * y_pred_f)
            score = (2. * intersection + self.smooth) / (
                        torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)
            return score

    def dice_loss(self, y_true, y_pred):
            loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
            return loss

    def log_cosh_dice_loss(self, y_true, y_pred):
            x = self.dice_loss(y_true, y_pred)
            return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
