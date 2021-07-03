#!/usr/bin/env python3

#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EfficientNet V2 models."""

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
)
from torch.nn import Dropout, Module



class EffHead(Module):
    """EfficientNet V2 head: 1x1, BN, AF, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, num_classes):
        super(EffHead, self).__init__()
        dropout_ratio = cfg.EN.DROPOUT_RATIO
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(w_out)
        self.conv_af = activation()
        self.avg_pool = gap2d(w_out)
        self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
        self.fc = linear(w_out, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if self.dropout else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, num_classes):
        cx = conv2d_cx(cx, w_in, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        cx = gap2d_cx(cx, w_out)
        cx = linear_cx(cx, w_out, num_classes, bias=True)
        return cx


class MBConv(Module):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out):
        # Expansion, kxk dwise, BN, AF, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = activation()
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(w_exp)
        self.dwise_af = activation()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = norm2d(w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.EN.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, cfg.EN.DC_RATIO)
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
        cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = conv2d_cx(cx, w_exp, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx

class FusedMBConv(Module):
    """Mobile inverted bottleneck block with SE."""
    """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""
    def __init__(self, w_in, exp_r, k, stride, se_r, w_out):
        # Expansion, kxk dwise, BN, AF, SE, 1x1, BN, skip_connection
        super().__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, k,stride=stride)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = activation()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = conv2d(w_exp, w_out, 1 if w_exp != w_in else k)
        self.lin_proj_bn = norm2d(w_out)
        self.lin_proj_af=activation()
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.se(f_x)
        f_x =self.lin_proj_bn(self.lin_proj(f_x))  if self.exp else self.lin_proj_af(self.lin_proj_bn(self.lin_proj(f_x)))
        if self.has_skip:
            if self.training and cfg.ENV2.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, cfg.ENV2.DC_RATIO)
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = conv2d_cx(cx, w_in, w_exp, k,stride=stride)
            cx = norm2d_cx(cx, w_exp)
        cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = conv2d_cx(cx, w_exp, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class EffV2Stage(Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out, d, block_type):
        super().__init__()
        for i in range(d):
            if block_type =="MB":
                block = MBConv(w_in, exp_r, k, stride, se_r, w_out)
            else:
                block = FusedMBConv(w_in, exp_r, k, stride, se_r, w_out)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out, d, block_type):
        for _ in range(d):
            if block_type=="MB":
                cx = MBConv.complexity(cx, w_in, exp_r, k, stride, se_r, w_out)
            else:
                cx=FusedMBConv.complexity(cx, w_in, exp_r, k, stride, se_r, w_out)
            stride, w_in = 1, w_out
        return cx


class StemIN(Module):
    """EfficientNet V2 stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


class EffNet_V2(Module):
    """EfficientNet V2 model."""

    @staticmethod
    def get_params():
        return {
            "sw": cfg.ENV2.STEM_W,
            "ds": cfg.ENV2.DEPTHS,
            "bt": cfg.ENV2.BLOCK_TYPE,
            "ws": cfg.ENV2.WIDTHS,
            "exp_rs": cfg.ENV2.EXP_RATIOS,
            "se_r": cfg.ENV2.SE_R,
            "ss": cfg.ENV2.STRIDES,
            "ks": cfg.ENV2.KERNELS,
            "hw": cfg.ENV2.HEAD_W,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super().__init__()
        p = EffNet_V2.get_params() if not params else params
        vs = ["sw", "ds", "bt", "ws", "exp_rs", "se_r", "ss", "ks", "hw", "nc"]
        sw, ds, b_types, ws, exp_rs, se_rs, ss, ks, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, b_types, ws, exp_rs, ss, ks,se_rs))
        self.stem = StemIN(3, sw)
        prev_w = sw
        for i, (d, b_type, w, exp_r, stride, k,se_r) in enumerate(stage_params):
            stage = EffV2Stage(prev_w, exp_r, k, stride, se_r, w, d, b_type)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = EffHead(prev_w, hw, nc)
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = EffNet_V2.get_params() if not params else params
        vs = ["sw", "ds", "bt", "ws", "exp_rs", "se_r", "ss", "ks", "hw", "nc"]
        sw, ds, b_types, ws, exp_rs, se_rs, ss, ks, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, b_types, ws, exp_rs, ss, ks,se_rs))
        cx = StemIN.complexity(cx, 3, sw)
        prev_w = sw
        for d, b_type, w, exp_r, stride, k,se_r in stage_params:
            cx = EffV2Stage.complexity(cx, prev_w, exp_r, k, stride, se_r, w, d,b_type)
            prev_w = w
        cx = EffHead.complexity(cx, prev_w, hw, nc)
        return cx
