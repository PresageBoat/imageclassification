#!/usr/bin/env python3


#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MobilenetV3 small models."""

from core.config import cfg
from models.blocks import (
    SE,
    conv2d,
    conv2d_cx,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
)
from torch.nn import Dropout, Module
import torch.nn as nn
import torch



def mb_activation(AF):
    """Helper for building a activation layer."""
    if AF=="RE":
        return nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
    elif AF=="HS":
        return torch.nn.Hardswish(inplace=cfg.MODEL.ACTIVATION_INPLACE)

class MBSHead(Module):
    """mobilenetv3-small head: 1x1, BN, AF, AvgPool, FC(Linear), AF, Dropout, FC(Linear)."""

    def __init__(self, w_in, w_out,head_w, num_classes):
        super(MBSHead, self).__init__()
        dropout_ratio = cfg.MBS.DROPOUT_RATIO
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(w_out)
        self.conv_af_hs = mb_activation('HS')
        self.avg_pool = gap2d(w_out)
        self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
        self.fc1 = linear(w_out, head_w, bias=True)
        self.fc2 = linear(head_w, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af_hs(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.conv_af_hs(x)
        x = self.dropout(x) if self.dropout else x
        x = self.fc2(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out,head_w, num_classes):
        cx = conv2d_cx(cx, w_in, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        cx = gap2d_cx(cx, w_out)
        cx = linear_cx(cx, w_out,head_w)
        cx = linear_cx(cx, head_w, num_classes, bias=True)
        return cx


class MBConv(Module):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, w_in, w_exp, k, stride, se_r, w_out,af):
        # Expansion, kxk dwise, BN, AF, SE, 1x1, BN, skip_connection
        super().__init__()
        self.exp = None
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = mb_activation(af)
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(w_exp)
        self.dwise_af = mb_activation(af)
        self.use_se= 1 if se_r>0 else 0
        if self.use_se:
            self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = norm2d(w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        if self.use_se:
            f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = conv2d_cx(cx, w_in, w_exp, 1)
            cx = norm2d_cx(cx, w_exp)
        cx = conv2d_cx(cx, w_exp, w_exp, k, stride=stride, groups=w_exp)
        cx = norm2d_cx(cx, w_exp)
        if se_r>0:
            cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = conv2d_cx(cx, w_exp, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class MBSStemIN(Module):
    """MobileNetV3 small stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super().__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = mb_activation('HS')

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx

class MBSStage(Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out, d):
        super().__init__()
        for i in range(d):
            block = MBConv(w_in, exp_r, k, stride, se_r, w_out)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out, d):
        for _ in range(d):
            cx = MBConv.complexity(cx, w_in, exp_r, k, stride, se_r, w_out)
            stride, w_in = 1, w_out
        return cx



class MobileNetV3_S(Module):
    """mobilenetv3 small model."""

    def __init__(self, params=None):
        super().__init__()
        self._construct_imagenet()
        self.apply(init_weights)
    
    def _construct_imagenet(self):
        head_w=cfg.MBS.HEAD_W
        se_r=cfg.MBS.SE_R
        self.stem=MBSStemIN(3,16)
        # w_in, w_exp, k, stride, se_r, w_out,af
        self.s1=MBConv(16,16,3,2,se_r,16,'RE')
        self.s2=MBConv(16,72,3,2,0,24,'RE')
        self.s3_1=MBConv(24,88,3,1,0,24,'RE')
        self.s3_2=MBConv(24,96,5,2,se_r,40,'HS')
        self.s4_1=MBConv(40,240,5,1,se_r,40,'HS')
        self.s4_2=MBConv(40,240,5,1,se_r,40,'HS')
        self.s4_3=MBConv(40,120,5,1,se_r,48,'HS')
        self.s4_4=MBConv(48,144,5,1,se_r,48,'HS')
        self.s4_5=MBConv(48,288,5,2,se_r,96,'HS')
        self.s5_1=MBConv(96,576,5,1,se_r,96,'HS')
        self.s5_2=MBConv(96,576,5,1,se_r,96,'HS')
        self.head=MBSHead(96,576,head_w,cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity (if you alter the model, make sure to update)."""
        head_w=cfg.MBS.HEAD_W
        se_r=cfg.MBS.SE_R
        cx=MBSStemIN.complexity(cx,3,16)
        cx=MBConv.complexity(cx,16,16,3,2,se_r,16)
        cx=MBConv.complexity(cx,16,72,3,2,0,24)
        cx=MBConv.complexity(cx,24,88,3,1,0,24)
        cx=MBConv.complexity(cx,24,96,5,2,se_r,40)
        cx=MBConv.complexity(cx,40,240,5,1,se_r,40)
        cx=MBConv.complexity(cx,40,240,5,1,se_r,40)
        cx=MBConv.complexity(cx,40,120,5,1,se_r,48)
        cx=MBConv.complexity(cx,48,144,5,1,se_r,48)
        cx=MBConv.complexity(cx,48,288,5,2,se_r,96)
        cx=MBConv.complexity(cx,96,576,5,1,se_r,48)
        cx=MBConv.complexity(cx,96,576,5,1,se_r,48)
        cx=MBSHead.complexity(cx,96,576,head_w,cfg.MODEL.NUM_CLASSES)
        return cx
