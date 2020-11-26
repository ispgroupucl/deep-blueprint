import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class ReconstructSegInterpolation(torch.nn.Module):
    """Reconstruct image by weighted interpolation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        img = x["image"]
        seg = x["segmentation"]
        mask = x["mask"]
        if len(img.shape) == 4:  # TODO : take into account batch
            conv = F.conv2d
            kernel = partial(gaussian_kernel, d=2)
        elif len(img.shape) == 5:
            conv = F.conv3d
            kernel = partial(gaussian_kernel, d=3)
        else:
            raise ValueError("Only 2D and 3D images work.")

        kernel_size = 9
        k = (
            kernel(size=kernel_size, sigma=10.0)
            .unsqueeze_(0)
            .unsqueeze_(0)
            .to(torch.double)
        )
        ones_kernel = torch.ones_like(k)
        recon_img = conv(img, k, padding=(kernel_size // 2))
        non_masked_num = conv(mask, ones_kernel, padding=(kernel_size // 2))
        non_masked_num[non_masked_num == 0] = 1
        print(non_masked_num.max(), non_masked_num.min())
        return recon_img * kernel_size * kernel_size * mask / non_masked_num


def gaussian_kernel(size=3, sigma=1.0, d=2):
    """Creates a gaussian kernel.
    
    Adapted from https://stackoverflow.com/a/43346070/3832318.
    """

    ax = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    sigma_square = torch.square(torch.tensor(sigma))

    if d == 2:
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / sigma_square)
    elif d == 3:
        xx, yy, zz = torch.meshgrid(ax, ax, ax)
        kernel = torch.exp(
            -0.5
            * (torch.square(xx) + torch.square(yy) + torch.square(zz))
            / sigma_square
        )
    else:
        raise ValueError("Only 2 and 3 dimensional kernels are supported.")
    return kernel / torch.sum(kernel)
