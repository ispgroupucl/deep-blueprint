import warnings

from .blocks import create_cba, get_all_blocks

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    """ Contains the layers used for the model
        if @all is set, its value will override @conv & @act
    """

    def __init__(self, conv=None, bn=None, act=None):
        self.conv = conv if conv is not None else nn.Conv2d

        # TODO: BN not supported via string yet
        self.bn = bn if bn is not None else FreezableBatchNorm2d
        self.act = act if act is not None else nn.LeakyReLU

        self.fc = nn.Linear

        self.conv_t = nn.ConvTranspose2d


class FreezableBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        """
            If @freeze_stats is set to True and the layer is frozen,

        """
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.frozen = False

    def train(self, mode=True):
        if self.frozen:
            super().train(False)
        else:
            super().train(mode)


class ConvBlock(nn.Module):
    """Modular block for unet-like type networks

    # Arguments:
        model_cfg:   the eventual quantization
        Block:       which block to use
        in_width:    the #input_channels
        width:       the #output_channels
        ksize:       the conv kernel size
        concatenate: 1 or 2 inputs during forward pass
        bn:          use or don't use normalization
        pool:        one of {max, up, None} to describe pooling
        drop:        amount of dropout to apply
        first:       first block or not
        concat_width: #channels of skip connection
        last_only:   if True only returns last layer
                     otherwise returns tuple: (block_output, last_layer)
    """

    def __init__(
        self,
        model_cfg,
        Block,
        in_width,
        width,
        ksize,
        concatenate=False,
        bn=True,
        pool=None,
        drop=(0, 0),
        first=False,
        conv_transpose=False,
        concat_width=None,
        last_only=False,
        se_ratio=0,
        dilation=1,
    ):
        super().__init__()
        # Concatenate from somwhere else (unet-like)
        self.concatenate = concatenate
        self.conv_transpose = conv_transpose
        self.last_only = last_only

        # Block
        dropout, sddrop = drop
        self.block = Block(
            model_cfg,
            in_width if not concatenate else in_width + concat_width,
            width,
            ksize,
            bn=bn,
            act=True,
            first=first,
            se_ratio=se_ratio,
            dilation=dilation,
            sddrop=sddrop,
        )
        self.drop = (
            nn.Dropout2d(dropout, inplace=False) if dropout > 0.0 else lambda xx: xx
        )

        # Pooling
        if pool not in ["up", "max", None]:
            warnings.warn("Pooling layer [%s] not supported" % pool)
            exit(-1)
        self.pool = pool
        self.pool_size = 2
        if pool == "up":
            upsampling = (
                nn.ConvTranspose2d(width, width, self.pool_size, stride=2)
                if self.conv_transpose
                else nn.UpsamplingBilinear2d(scale_factor=self.pool_size)
            )
            # TODO: check if doing depthwise wouldn't be a solution
            CBA = create_cba(model_cfg)
            self.up_block = nn.Sequential(
                upsampling, CBA(width, width, self.pool_size, bn=bn, act=False)
            )

    def forward(self, nx):
        # Concat
        if self.concatenate:
            nx = torch.cat(nx, dim=1)
        # Block
        nx = conv_out = self.block(nx)
        nx = self.drop(nx)
        # Pooling
        if self.pool is not None:
            if self.pool == "max":
                nx = F.max_pool2d(nx, self.pool_size)
            else:  # pool == "up"
                nx = self.up_block(nx)
        return (conv_out, nx) if not self.last_only else nx


class SubIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, nx):
        if type(nx) is tuple:
            return nx[0]
        return nx
