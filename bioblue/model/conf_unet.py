import warnings
import itertools

from .blocks import create_cba, get_all_blocks
from .model_wrapper import ModelWrapper
from .utils import ConvBlock, SubIdentity
from hydra.utils import instantiate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfUnet(ModelWrapper):
    def __init__(
        self,
        model_cfg,
        block="residual",
        input_size=(1, 512, 512),
        classes=("fg",),
        last_act="linear",
        conv_transpose=False,
        bn=False,
        architecture=None,
        big_drop=0.0,
        small_drop=0.0,
        sddrop=0.0,
        se_ratio=0.0,
        input_format=None,
        output_format=None,
        version=1,
        multi_input=False,
        features_only=False,
        kill=None,
    ):
        if hasattr(model_cfg, "_target_"):
            model_cfg = instantiate(model_cfg)

        super().__init__(
            model_cfg=model_cfg,
            classes=classes,
            last_act=last_act,
            output_format=output_format,
            input_format=input_format,
            input_size=input_size,
        )

        self.classes = classes
        self.n_classes = len(classes) + 1
        self.conv_transpose = conv_transpose
        self.multi_input = multi_input
        self.features_only = features_only
        self.kill = kill or set()

        # Parse the network's architecture
        if architecture is None:
            architecture = {
                "first": 32,
                "enc": {"width": [16, 32, 48, 96], "repeat": [2, 3, 3, 4]},
                "dec": {"width": [48, 32, 32], "repeat": [2, 2, 1]},
            }
        arch = architecture
        assert (
            len({"first", "enc", "dec"} - {*list(arch.keys())}) == 0
        ), "Missing keys: Need enc, dec, first"
        assert (
            len({"repeat", "width"} - {*list(arch["enc"].keys())}) == 0
        ), "Missing keys enc: Need width, repeat"
        assert (
            len({"repeat", "width"} - {*list(arch["dec"].keys())}) == 0
        ), "Missing keys dec: Need width, repeat"
        assert len(arch["enc"]["repeat"]) == len(
            arch["enc"]["width"]
        ), "Mismatched dimensions"
        assert len(arch["dec"]["repeat"]) == len(
            arch["dec"]["width"]
        ), "Mismatched dimensions"
        self.arch = arch
        arch["width"] = arch["enc"]["width"] + arch["dec"]["width"]
        arch_enc_len = len(arch["enc"]["width"])
        arch_dec_len = len(arch["dec"]["width"])

        # Generate Basic building block & Bigger block
        CBA = self.CBA

        all_blocks = get_all_blocks()
        if type(block) is not list:
            block = [block, block]

        blocks = {}
        for bl, name in zip(block, ["enc", "dec"]):
            if bl not in all_blocks:
                raise ValueError("Block " + bl + " is not a valid block option")
            blocks[name] = all_blocks[bl]

        # Encoder
        bw = arch["first"]
        if self.multi_input:
            self.multi_input_widening = []
        self.input_process = {}
        for key, in_size in zip(self.input_format, self.input_format_sizes):
            self.input_process[key] = CBA(in_size, bw, 3, bn=bn, act=True)
        self.input_process = nn.ModuleDict(self.input_process)

        self.encoder = []
        if len(self.input_format) == 1:
            self.encoder.append(nn.Identity())
        else:
            self.encoder.append(CBA(bw, bw, 3, bn=bn, act=True))
        prev_bw = bw
        skips_bw = []
        for i, repeat_block in enumerate(self.arch["enc"]["repeat"]):
            is_last = i + 1 == arch_enc_len
            if i > 0 and self.multi_input:
                self.multi_input_widening.append(
                    CBA(arch["first"], prev_bw, 1, bias=False)
                )
            for j in range(repeat_block):
                if version == 1:
                    new_bw = (
                        arch["width"][i]
                        if j + 1 < repeat_block
                        else arch["width"][i + 1]
                    )
                else:  # version==2
                    new_bw = arch["width"][i]
                pool = "max" if j + 1 == repeat_block and not is_last else None
                pool = "up" if j + 1 == repeat_block and is_last else pool
                drop = small_drop if (not is_last) or j + 1 < repeat_block else big_drop

                self.encoder.append(
                    ConvBlock(
                        model_cfg,
                        blocks["enc"],
                        prev_bw,
                        new_bw,
                        3,
                        bn=bn,
                        pool=pool,
                        conv_transpose=self.conv_transpose,
                        drop=(drop, sddrop),
                        se_ratio=se_ratio,
                        first=(i == 0),
                    )
                )
                prev_bw = new_bw
            skips_bw.append(prev_bw)
        self.encoder = nn.ModuleList(self.encoder)
        if self.multi_input:
            self.multi_input_widening = nn.ModuleList(self.multi_input_widening)

        # Decoders (Classif, Pif, Paf...)
        skips_bw.reverse()  # Reverse for easier indexing

        def get_decoder(prev_bw, bw):
            decoder = []
            for i, repeat_block in enumerate(self.arch["dec"]["repeat"]):
                is_last = i + 1 == arch_dec_len
                for j in range(repeat_block):
                    if version == 1:
                        new_bw = (
                            arch["width"][arch_enc_len + i]
                            if j + 1 < repeat_block or is_last
                            else arch["width"][arch_enc_len + i + 1]
                        )
                    else:  # version==2
                        new_bw = arch["width"][arch_enc_len + i]

                    pool = "up" if not is_last and j + 1 == repeat_block else None
                    has_skip = j == 0
                    concat_width = skips_bw[i + 1] if has_skip else None

                    decoder.append(
                        ConvBlock(
                            model_cfg,
                            blocks["dec"],
                            prev_bw,
                            new_bw,
                            3,
                            bn=bn,
                            concatenate=has_skip,
                            concat_width=concat_width,
                            pool=pool,
                            conv_transpose=self.conv_transpose,
                            drop=(small_drop, sddrop),
                            se_ratio=se_ratio,
                            last_only=True,
                        )
                    )
                    prev_bw = new_bw
            return decoder, prev_bw

        enc_bw = bw
        enc_prev_bw = prev_bw
        decoder_types = {
            "segmentation": lambda: get_decoder(enc_prev_bw, enc_bw / 2),
            "keypoints": lambda: get_decoder(enc_prev_bw, enc_bw / 2),
            "class": lambda: (
                [SubIdentity()]
                * (np.sum(arch["dec"]["repeat"])),  # FIXME: not really efficient
                enc_prev_bw,
            ),
        }
        self.decoders = []
        decoders_prev_bw = []
        for out_class in self.output_format:
            decoder, bw_dec = decoder_types[out_class]()
            self.decoders.append(nn.ModuleList(decoder))
            decoders_prev_bw.append(bw_dec)
        self.decoders = nn.ModuleList(self.decoders)

        # Tops
        self.tops = self.make_tops(decoders_prev_bw)

    def forward(self, nx):
        all_inputs = []
        # Divide different inputs
        d_scale = 0
        tot_scale = 0
        scale = len(self.input_format)
        for dtype in self.input_format:
            processed_dtype = self.input_process[dtype](nx[dtype])
            tot_scale = tot_scale + torch.mean(
                processed_dtype, dim=(2, 3), keepdim=True
            )
            if dtype in self.kill:
                scale -= 1
                continue
            d_scale = d_scale + torch.mean(processed_dtype, dim=(2, 3), keepdim=True)
            all_inputs.append(processed_dtype)
        if self.features_only:
            return (torch.transpose(torch.stack(all_inputs), 0, 1),)
        # nx = all_inputs = (len(self.input_format)/scale) * torch.stack(all_inputs).sum(dim=0)
        nx = all_inputs = (tot_scale - d_scale) + torch.stack(all_inputs).sum(dim=0)
        # nx = all_inputs = torch.stack(all_inputs).sum(dim=0)

        # Encoder
        nx = self.encoder[0](nx)
        encs = [None] * (len(self.arch["enc"]["repeat"]))  # save skip connections
        ind = 1
        for i, repeat_block in enumerate(self.arch["enc"]["repeat"]):
            if i > 0 and self.multi_input:
                all_inputs = F.avg_pool2d(all_inputs, (2, 2))
                nx = nx + self.multi_input_widening[i - 1](all_inputs)
            for j in range(repeat_block):
                encs[i], nx = self.encoder[ind](nx)
                ind += 1

        enc_nx = nx
        encs.reverse()  # reverse for easier indexing

        # Decoders
        decs_nx = []
        for decoder in self.decoders:
            nx = enc_nx
            ind = 0
            for i, repeat_block in enumerate(self.arch["dec"]["repeat"]):
                for j in range(repeat_block):
                    nx = (nx, encs[i + 1]) if j == 0 else nx
                    nx = decoder[ind](nx)
                    ind += 1
            decs_nx.append(nx)

        if self.features_only:
            nd = self.tops(decs_nx)[0]
            return (torch.cat((nx, nd), axis=1),)

        # Tops
        return self.tops(decs_nx)
