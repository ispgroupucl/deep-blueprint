from turtle import forward
from .blocks import create_cba, get_all_blocks, ClassificationOutput
from .model_wrapper import ModelWrapper
from .utils import ConvBlock, SubIdentity
from hydra.utils import instantiate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def flattened_size(w, k=3, s=1, p=0, m=True):
    """
    Returns the right size of the flattened tensor after
        convolutional transformation
    :param w: width of image
    :param k: kernel size
    :param s: stride
    :param p: padding
    :param m: max pooling (bool)
    :return: proper shape and params: use x * x * previous_out_channels

    Example:
    r = flatten(*flatten(*flatten(w=100, k=3, s=1, p=0, m=True)))[0]
    self.fc1 = nn.Linear(r*r*128, 1024)
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m


class ConfResNet(ModelWrapper):
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
        # print('self.classes', self.classes)

        self.conv_transpose = conv_transpose
        self.multi_input = multi_input
        self.features_only = features_only
        self.kill = kill or set()

        # Generate Basic building block & Bigger block
        CBA = self.CBA

        all_blocks = get_all_blocks()
        if type(block) is not list:
            block = [block, block]


        arch = architecture
        self.arch = arch
        bw = arch["first"]

        if self.multi_input:
            self.multi_input_widening = []
        self.input_process = {}
        
        self.fcn_input_format = [ inp for inp in self.input_format if inp != 'image']
        self.input_format = [inp for inp in self.input_format if inp == 'image']

        # Create initial conv2d net: InputImage -> architecture.first
        for key, in_size in zip(self.input_format, self.input_format_sizes):
            self.input_process[key] = CBA(in_size, bw, 3, bn=bn, act=True)
        self.input_process = nn.ModuleDict(self.input_process)


        self.blocks = []
        if len(self.input_format) == 1:
            self.blocks.append(nn.Identity())
        else:
            self.blocks.append(CBA(bw, bw, 3, bn=bn, act=True))
        prev_bw = bw
        skips_bw = []

        for i, repeat_block in enumerate(self.arch["num_resnet_blocks"]):

            # print('repeat_block', repeat_block)

            is_last = i + 1 == len(self.arch["num_resnet_blocks"])
            
            for j  in range(repeat_block):
                new_bw = (
                    arch["block_width"][i]
                    # if j + 1 < repeat_block
                    # else arch["block_width"][i + 1]
                )
                pool = "max" if j + 1 == repeat_block and not is_last else None
                pool = None if j + 1 == repeat_block and is_last else pool
                drop = small_drop if (not is_last) or j + 1 < repeat_block else big_drop

                # print(f'\t {j}: {prev_bw} -> {new_bw} -> {pool}')

                self.blocks.append(
                    ConvBlock(
                        model_cfg,
                        all_blocks["residual"],
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
                # print(f'\t {j}')
            skips_bw.append(prev_bw)
        self.blocks = nn.ModuleList(self.blocks)

        # print('finished Resnet Blocks')
        
        # self.output_net = ClassificationOutput(model_cfg,
        #                                         prev_bw,
        #                                         len(classes), 
        #                                         'softmax')

        # self.output_net = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Flatten(),
        #     nn.Linear(prev_bw, self.n_classes)
        # )

        self.output_cnn_net = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )


        # output_cnn_net_size = 0 # flattened_size(,k=3,s=1)
        output_cnn_net_size = self.arch["cnn_input_size"] // 2**(len(self.arch["block_width"])-1)
        output_cnn_net_size = (output_cnn_net_size**2)*self.arch["block_width"][-1]
        # print("output_cnn_net_size", output_cnn_net_size)

        initial_fcn_layer = nn.Linear(output_cnn_net_size + len(self.fcn_input_format),self.arch['fcn_width'])
        hidden_fcn_layers = [ nn.Sequential(
                                    nn.Linear(self.arch['fcn_width'],self.arch['fcn_width']),
                                    nn.ReLU()
                                )
                            for i in range(arch["fcn_num_hidden"])]
        fcn2class_layer = nn.Linear(self.arch['fcn_width'],self.n_classes)

        self.classifier_fcn = nn.Sequential(
            initial_fcn_layer,
            nn.ReLU(),
            *hidden_fcn_layers,
            fcn2class_layer
        )
        # print('finished final output_block')
        
        # prev_bw = prev_bw

        # self.output_fcn= nn.Sequential(
        #     nn.Linear(prev_bw, self.n_classes)
        # )

        # print('finished final output_block')

    def forward(self,nx):
        # print('conf_resnet', [(key , nx[key].shape) for  key in nx.keys()])
        # print(nx['image'].shape)
        # print(self.input_process)

        all_inputs = []

        for dtype in self.input_format:
            
            processed_dtype = self.input_process[dtype](nx[dtype])

            all_inputs.append(processed_dtype)

        # print("AFTER INPUT_PROCESS")
        # print([ (key, nx[key].shape) for key in nx])

        nx_cnn = all_inputs = torch.stack(all_inputs).sum(dim=0)
        # print(f"AFTER input_process: {nx_cnn.shape}")
        
        # print(nx.shape)
        # print(self.blocks[0])
        nx_cnn = self.blocks[0](nx_cnn) #identity block   
        # print(f"AFTER Identity: {nx_cnn.shape}")

        for i, block in enumerate(self.blocks[1:]):
            # print(i)
            # print(block)
            # print( block(nx_cnn))
            _, nx_cnn =  block(nx_cnn)

            # print(f"AFTER Block {i}: {nx_cnn.shape}")

        encoded_nx_cnn = nx_cnn

        # print('encoded_nx_cnn', encoded_nx_cnn.shape)

        flattened_cnn_output = self.output_cnn_net(encoded_nx_cnn)

        # print('flattened_cnn_output', flattened_cnn_output.shape)

        fcn_inputs = []
        for dtype in self.fcn_input_format:
            fcn_inputs.append(torch.squeeze(nx[dtype],2))
        fcn_inputs = torch.cat(fcn_inputs, 1 )
        # print("fcn_inputs: ",fcn_inputs.shape)
        fcn_inputs = torch.cat([fcn_inputs, flattened_cnn_output], 1 )
        # print("FULL_fcn_inputs: ",fcn_inputs.shape)

        # raise NotImplementedError

        fc_nx = self.classifier_fcn(fcn_inputs)
        # fc_nx = self.output_net(encoded_nx_cnn)

        # # print("fc_nx",  fc_nx.shape)
        return fc_nx
        





        
