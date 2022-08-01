import torch.nn as nn
import torch.nn.functional as F
import torch


class StochasticDepthDropout(nn.Module):
    """Dropout for RESIDUAL blocks only
                -> stochastic depth: https://arxiv.org/pdf/1603.09382.pdf
        Implementation taken from official Efficientnet github

    Args:
        nx (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of dropped sample.
    Returns:
        output: Output where some samples will be dropped.
    """

    def __init__(self, p=0.2):
        super().__init__()
        assert p >= 0 and p <= 1, "p must be in range of [0,1]"
        self.p = p

    def forward(self, nx):
        if not self.training:
            return nx

        batch_size = nx.shape[0]
        keep_prob = 1 - self.p

        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand(
            [batch_size, 1, 1, 1], dtype=nx.dtype, device=nx.device
        )

        nx = nx / keep_prob * torch.floor(random_tensor)
        return nx


def _get_module_size(module):
    size = 0
    for _ in module.modules():
        size += 1
    return size


def get_all_blocks():
    """
    # Returns:
        a dictionary containing all defined conv blocks
    """
    return {
        "mobile": MobileBlock,
        "mobile6": lambda *args, **kwargs: MobileBlock(*args, **{**kwargs, "widen": 6}),
        "residual": ResidualBlock,
        "residual2": lambda *args, **kwargs: ResidualBlock(
            *args, **{**kwargs, "v2": True}
        ),
        "double": DoubleBlock,
        "simple": SimpleBlock,
    }


def get_last_act(last_act):
    """
    # Returns:
        The activation function corresponding to last_act
    """
    return {
        "sigmoid": torch.sigmoid,
        "softmax": lambda xx: F.softmax(xx, dim=1),
        "relu": F.relu,
        "linear": lambda xx: xx,  # hack for pytorch
    }.get(last_act, None)


def create_cba(model_cfg):
    """Create the basic building of all networks.

    # Arguments:
        _Conv:          nn.Conv2d by default but could also be a quantized version
        _Normalization: BatchNorm, GroupNorm
        _Activation:    ReLU, Sigmoid, Tanh etc.
    """
    _Conv = model_cfg.conv
    _Normalization = model_cfg.bn
    _Activation = model_cfg.act
    _Zeropad = model_cfg.zeropad
    dim = model_cfg.dim

    class BasicBlock(nn.Module):
        """Basic building block of all networks.

        # Arguments
            in_width:     number of input channels
            width:        number of output channels
            ksize:        size of kernel (tuple or integer)
            bn, act:      bools to enable/disable layer
            reverse:      CBA or BAC
            bias:         add/remove conv. bias
            padding_type: one of 'zeros', 'ones' or 'reflect'
            stride:       integer
            depthwise:    use a seperable version or not (cfr. MobileNet)
        """

        def __init__(
            self,
            in_width,
            width,
            ksize,
            bn=True,
            act=True,
            reverse=False,
            bias=True,
            padding_type="zeros",
            stride=1,
            depthwise=False,
            dilation=1,
        ):
            super(BasicBlock, self).__init__()
            if depthwise:
                assert (
                    in_width == width
                ), "Input & output width should be equal for depthwise convolutions"
            self.reverse = reverse
            bn_width = in_width if reverse else width

            bn = _Normalization(bn_width) if bn else None
            act = _Activation() if act else None
            if hasattr(act, "inplace"):
                act.inplace = True

            padding = ksize // 2
            conv = _Conv(
                in_width,
                width,
                ksize,
                stride=stride,
                dilation=dilation,
                padding=padding if ksize % 2 != 0 else 0,
                bias=bias,
                padding_mode=padding_type,
                groups=in_width if depthwise else 1,
            )
            if ksize % 2 == 0 and padding > 0:
                conv = nn.Sequential(
                    _Zeropad(
                        dim * (padding, padding - 1)
                    ),  # TODO: check if 0's should not be padding-1
                    conv,
                )

            block = [conv, bn, act] if not reverse else [bn, act, conv]
            block = [elem for elem in block if elem is not None]
            self.cba = nn.Sequential(*block)

        def forward(self, nx):
            return self.cba(nx)

    return BasicBlock


class DoubleBlock(nn.Module):
    def __init__(
        self, model_cfg, in_width, width, ksize, stride=1, bn=True, act=True, **kwargs
    ):
        super(DoubleBlock, self).__init__()
        CBA = create_cba(model_cfg)
        self.block1 = CBA(in_width, width, ksize, bn=bn, act=act, stride=stride)
        self.block2 = CBA(width, width, ksize, bn=bn, act=act)

    def forward(self, nx):
        return self.block2(self.block1(nx))


class SimpleBlock(nn.Module):
    def __init__(
        self, model_cfg, in_width, width, ksize, stride=1, bn=True, act=True, **kwargs
    ):
        super(SimpleBlock, self).__init__()
        CBA = create_cba(model_cfg)
        self.block = CBA(in_width, width, ksize, bn=bn, act=act, stride=stride)

    def forward(self, nx):
        return self.block(nx)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        model_cfg,
        in_width,
        width,
        ksize,
        stride=1,
        bn=True,
        act=True,
        v2=False,
        se_ratio=0.0,
        sddrop=0,
        **kwargs
    ):
        super().__init__()
        CBA = create_cba(model_cfg)
        self.has_final_act = (not v2) and act
        if self.has_final_act:
            self.final_act = model_cfg.act()
        self.block1 = CBA(
            in_width, width, ksize, bn=bn, act=act, reverse=v2, stride=stride
        )
        self.block2 = CBA(
            width, width, ksize, bn=bn, act=act if v2 else False, reverse=v2
        )

        self.sdd = StochasticDepthDropout(p=sddrop) if sddrop > 0 else lambda xx: xx

        self.id_block = (
            CBA(in_width, width, 1, bn=bn, act=False, stride=stride)
            if (in_width != width) or stride > 1
            else lambda xx: xx
        )
        self.has_se = se_ratio > 0
        if self.has_se:
            self.se = SEModule(model_cfg, width, se_ratio)

    def forward(self, nx):
        _in = nx
        shortcut = self.id_block(_in)
        nx = self.block2(self.block1(self.sdd(nx)))
        if self.has_se:
            nx = self.se(nx)
        nx = nx + shortcut
        if self.has_final_act:
            nx = self.final_act(nx)
        return nx


class MobileBlock(nn.Module):
    def __init__(
        self,
        model_cfg,
        in_width,
        width,
        ksize,
        bn=True,
        act=True,
        stride=1,
        first=False,
        residual=True,
        widen=2,
        se_ratio=0,
        sddrop=0,
        **kwargs
    ):
        super(MobileBlock, self).__init__()
        CBA = create_cba(model_cfg)
        mid_width = int(width * widen)
        self.residual = residual

        self.s_conv = nn.Sequential(
            CBA(
                in_width,
                mid_width,
                1,
                bn=False if first else bn,
                act=False,
                reverse=True,
            ),
            CBA(
                mid_width,
                mid_width,
                ksize,
                bn=bn,
                act=act,
                reverse=True,
                depthwise=True,
                stride=stride,
            ),
            CBA(mid_width, width, 1, bn=bn, act=False, reverse=True),
        )
        if se_ratio > 0:
            self.se = SEModule(model_cfg, width, se_ratio)
        self.has_se = se_ratio > 0
        self.last_act = model_cfg.act()
        self.sdd = StochasticDepthDropout(p=sddrop) if sddrop > 0 else lambda xx: xx

        if self.residual and (in_width != width or stride > 1):
            self.residual = False

    def forward(self, nx):
        if self.residual:
            _in = nx
        nx = self.s_conv(self.sdd(nx))
        if self.has_se:
            nx = self.se(nx)
        if self.residual:
            nx = nx + _in
        return self.last_act(nx)


class ClassificationOutput(nn.Module):
    def __init__(self, model_cfg, in_width, n_classes, last_act):
        super(ClassificationOutput, self).__init__()
        CBA = create_cba(model_cfg)
        self.act = get_last_act(last_act)
        self.block = CBA(in_width, n_classes, 1, bn=False, act=False)

    def forward(self, nx):
        return self.act(self.block(F.relu(nx)))


class SEModule(nn.Module):
    def __init__(self, model_cfg, width, ratio):
        super().__init__()

        mid_width = max(1, int(width / ratio))
        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            model_cfg.conv(width, mid_width, 1),  # Bias is intentional
            model_cfg.act(),
            model_cfg.conv(mid_width, width, 1),
            nn.Sigmoid(),  # TODO: either add to model_cfg or use hardSigm
        )

    def forward(self, nx):
        _in = nx
        nx = self.excite(nx)
        return nx * _in
