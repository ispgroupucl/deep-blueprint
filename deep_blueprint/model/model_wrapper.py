from typing import Sequence
import warnings
from .blocks import get_last_act, create_cba

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools


class ModelWrapper(nn.Module):
    def __init__(
        self,
        model_cfg,
        classes=None,
        last_act="linear",
        input_format=None,
        output_format=None,
        input_size=None,
        repeat_outputs=None,
    ):
        super().__init__()

        # Default input-output formats
        if input_format is None:
            input_format = ["image"]

        if not isinstance(input_format, Sequence):
            input_format = [input_format]
        inf = [x.split("_") for x in input_format]
        self.input_format_sizes = []
        self.input_format = []
        for x in input_format:
            try:
                sx = x.split("_")
                bw = int(x[-1])
                x_dtype = "_".join(sx[:-1])
            except ValueError:
                bw = 1 if input_size is None else input_size[0]
                x_dtype = x
            finally:
                self.input_format_sizes.append(bw)
                self.input_format.append(x_dtype)

        if output_format is None:
            output_format = ["segmentation"]

        inference_output_format = output_format
        if repeat_outputs is not None:
            output_format = []
            for of in inference_output_format:
                if "class" not in of:
                    output_format += [of] * repeat_outputs
                else:
                    output_format.append(of)
        self.n_classes = len(classes) + 1
        self.output_format = output_format
        self.inference_output_format = output_format

        # Generate Basic building block & Bigger block
        self.CBA = create_cba(model_cfg)

        # Get final activation layer
        self.last_act = get_last_act(last_act)
        self.outputs = self.output_format

    def make_tops(self, prev_bw_list, upscale_list=None):
        """
            - prev_bw_list: is a list of the previous widths (aka #FM)
                            MUST have the same size as self.output_format
            - upscale_list: is a list of the amount of upscaling needed
                            MUST have the same size as self.output_format
             
        """
        assert len(prev_bw_list) == len(self.output_format)
        if upscale_list is None:
            upscale_list = [1] * len(self.output_format)
        assert len(upscale_list) == len(self.output_format)

        CBA = self.CBA
        # Tops

        self._tops = []
        for i, (out_class, prev_bw, up_ratio) in enumerate(
            zip(self.output_format, prev_bw_list, upscale_list)
        ):
            top = nn.Sequential(
                nn.ReLU(),
                CBA(prev_bw, self.n_classes, 1, bn=False, act=False),
                *(
                    [nn.UpsamplingBilinear2d(scale_factor=up_ratio)]
                    if up_ratio > 1
                    else []
                )
            )
            self._tops.append(top)
        self._tops = nn.ModuleList(self._tops)
        return self._forward_tops

    def _forward_tops(self, decs_nx_list):
        # Tops
        final_nx = tuple()
        for top, nx, out_class in zip(self._tops, decs_nx_list, self.output_format):
            out_type = out_class
            nx = self.last_act(top(nx))
            if out_type in {"class"}:
                nx = nx.squeeze_(-1).squeeze_(-1)
            elif out_type in {"segmentation"}:  # TODO: check if really necessary
                # Should we add keypoints too??
                if self.n_classes == 1:
                    nx = nx.squeeze_(1)
            final_nx += (nx,)

        return final_nx
