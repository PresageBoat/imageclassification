#!/usr/bin/env python3

#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ShuffleNet V2 models."""

from core.config import cfg
from models.blocks import (
    SE,
    activation,
    conv2d,
    conv2d_cx,
    drop_connect,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    pool2d,
    pool2d_cx,
)
from torch.nn import Dropout, Module
from torch import Tensor
import torch


class ShuffleStemIN(Module):
    """ShuffleNet v2 stem for ImageNet: 3x3, BN, AF, maxpool."""

    def __init__(self, w_in, w_out):
        super().__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.maxpool= pool2d(w_out,3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx,w_out,3, stride=2)
        return cx



class ShuffleHead(Module):
    """EfficientNet head: 1x1, BN, AF, FC"""

    def __init__(self, w_in, w_out, num_classes):
        super().__init__()
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(w_out)
        self.conv_af = activation()
        self.avg_pool = gap2d(w_out)
        self.fc = linear(w_out, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        #global pool
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        #x = x.mean([2, 3])  # globalpool
        f_x = self.fc(x)
        return f_x

    @staticmethod
    def complexity(cx, w_in, w_out, num_classes):
        cx = conv2d_cx(cx, w_in, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        cx = gap2d_cx(cx, w_out)
        cx = linear_cx(cx, w_out, num_classes, bias=True)
        return cx


class InvertedResidual(Module):
    def __init__(self, w_in, w_out,stride):
        super().__init__()
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features=w_out//2
        assert (self.stride != 1) or (w_in == branch_features << 1)
        if self.stride > 1:
            self.branch1=torch.nn.Sequential(
                self.depthwise_conv(w_in, w_in, kernel_size=3, stride=self.stride, padding=1),
                norm2d(w_in),
                conv2d(w_in,branch_features, kernel_size=1, stride=1, bias=False),
                norm2d(branch_features),
                activation(),
            )
        
        self.branch2 = torch.nn.Sequential(
            conv2d(w_in if(self.stride>1) else branch_features,branch_features,1,stride=1),
            norm2d(branch_features),
            activation(),
            self.depthwise_conv(branch_features,branch_features,3,stride=self.stride),
            conv2d(branch_features,branch_features,1,stride=1),
            norm2d(branch_features),
            activation(),
        )

    @staticmethod
    def depthwise_conv(self,w_in,w_out,k,stride):
        return conv2d(w_in, w_out,k, stride=stride,groups=w_in)

    @staticmethod    
    def channel_shuffle(x: Tensor, groups: int) -> Tensor:
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups,
                channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    
    def forward(self,x:Tensor):
        if self.stride==1:
            x1,x2=x.chunk(2,dim=1)
            f_x = torch.cat((x1,self.branch2(x2)),dim=1)
        else:
            f_x=torch.cat((self.branch1(x),self.branch2(x)),dim=1)
        f_x=self.channel_shuffle(f_x,2)
        return f_x
    
    @staticmethod
    def complexity(cx, w_in, w_out, stride):
        branch_features=w_out//2
        if stride > 1:
            cx=conv2d_cx(cx,w_in, w_in, kernel_size=3, stride=stride, padding=1)
            cx=norm2d_cx(cx,w_in)
            cx=conv2d_cx(cx,w_in,branch_features, kernel_size=1, stride=1, bias=False)
            cx=conv2d_cx(cx,branch_features)
        cx=conv2d_cx(cx,w_in if(stride>1) else branch_features,branch_features,1,stride=1),
        cx=norm2d_cx(cx,branch_features),
        cx=conv2d_cx(cx,branch_features,branch_features,3,stride=stride),
        cx=conv2d_cx(cx,branch_features,branch_features,1,stride=1),
        cx=norm2d_cx(cx,branch_features),
        return cx



class ShuffleStage(Module):
    """ShuffleNet v2 stage."""

    def __init__(self, w_in, w_out, stride, d):
        super().__init__()
        for i in range(d):
            block = InvertedResidual(w_in, w_out, stride)
            self.add_module("b{}".format(i + 1), block)
            stride,w_in=1,w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d):
        for _ in range(d):
            cx = InvertedResidual.complexity(cx, w_in, w_out, stride)
            stride,w_in=1,w_out

        return cx



class ShuffleNet_V2(Module):
    """ShuffleNet v2 model."""

    @staticmethod
    def get_params():
        return {
            "sw": cfg.SN.STEM_W,
            "ds": cfg.SN.DEPTHS,
            "ws": cfg.SN.WIDTHS,
            "ss": cfg.EN.STRIDES,
            "hw": cfg.EN.HEAD_W,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super().__init__()
        p = ShuffleNet_V2.get_params() if not params else params
        vs = ["sw", "ds", "ws", "ss", "hw", "nc"]
        sw, ds, ws, ss, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, ss))
        self.stem = ShuffleStemIN(3, sw)
        prev_w = sw
        for i, (d, w, stride) in enumerate(stage_params):
            stage = ShuffleStage(prev_w, w, stride, d)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = ShuffleHead(prev_w, hw, nc)
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = ShuffleNet_V2.get_params() if not params else params
        vs = ["sw", "ds", "ws","ss",  "hw", "nc"]
        sw, ds, ws, ss, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, ss))
        cx = ShuffleStemIN.complexity(cx, 3, sw)
        prev_w = sw
        for d, w, stride in stage_params:
            cx = ShuffleStage.complexity(cx, prev_w, w, stride, d)
            prev_w = w
        cx = ShuffleHead.complexity(cx, prev_w, hw, nc)
        return cx
