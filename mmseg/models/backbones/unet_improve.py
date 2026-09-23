# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import math
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch.nn.init import trunc_normal_
from mmseg.registry import MODELS
import torch.nn.functional as F
from mmseg.models.utils.wavelet import _ScaleModule, create_wavelet_filter, wavelet_transform, inverse_wavelet_transform


class ECA(nn.Module):
    def __init__(self, in_channel, kernel_size=5):
        super().__init__()
        self.adaptive = nn.AdaptiveAvgPool2d(1)
        self.k_size = kernel_size
        self.conv_adaptive = nn.Conv1d(in_channel, in_channel, kernel_size=self.k_size, bias=False, groups=in_channel)

    def forward(self, x):
        adaptive = self.adaptive(x)
        adaptive = nn.functional.unfold(adaptive.transpose(-1, -3),
                                        kernel_size=(1, self.k_size),
                                        padding=(0, (self.k_size - 1) // 2))
        x = self.conv_adaptive(adaptive.transpose(-1, -2)).unsqueeze(-1).sigmoid() * x
        return x


import torch.nn as nn
import torch


class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, scaling=16):  # scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class CBAM_Attention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAM_Attention, self).__init__()
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
class fusionBlock(nn.Module):
    """MDCF: multi-scale difference convolution fusion.

    Fuses a skip-connection feature ``x1`` with the upsampled decoder feature
    ``x2``.  A shared 3x3 branch produces the common component ``combine``;
    the two residual differences (``x1 - combine`` and ``x2 - combine``) are
    then enhanced by two parallel depth-wise convolutions each.

    Args:
        dim_in (int): Number of input and output channels.
        kernel_sizes (tuple[int, int]): Kernel sizes of the two depth-wise
            difference branches.  Defaults to ``(5, 7)`` (the paper setting).
            ``(3, 3)``, ``(5, 5)``, ``(7, 7)``, ``(9, 9)`` and ``(3, 5)`` are
            the variants used in the kernel-size ablation study.
    """

    def __init__(self, dim_in, kernel_sizes=(5, 7)):
        super(fusionBlock, self).__init__()
        k1, k2 = kernel_sizes
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim_in, dim_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
        )

        t = int(abs((math.log(dim_in, 2) + 1) / 2))
        kernel_size = t if t % 2 else t + 1
        self.eca1 = ECA(in_channel=dim_in, kernel_size=kernel_size)
        self.eca2 = ECA(in_channel=dim_in, kernel_size=kernel_size)

        self.conv_1_1 = nn.Conv2d(dim_in, dim_in, kernel_size=k1, padding=k1 // 2, groups=dim_in)
        self.conv_1_2 = nn.Conv2d(dim_in, dim_in, kernel_size=k2, padding=k2 // 2, groups=dim_in)

        self.conv_2_1 = nn.Conv2d(dim_in, dim_in, kernel_size=k1, padding=k1 // 2, groups=dim_in)
        self.conv_2_2 = nn.Conv2d(dim_in, dim_in, kernel_size=k2, padding=k2 // 2, groups=dim_in)

    def forward(self, x1, x2):
        x1 = self.eca1(x1)
        x2 = self.eca2(x2)

        combine = self.conv(torch.cat([x1, x2], dim=1))

        x1_diff = x1 - combine
        x2_diff = x2 - combine
        x1_diff = self.conv_1_1(x1_diff) + self.conv_1_2(x1_diff)
        x2_diff = self.conv_2_1(x2_diff) + self.conv_2_2(x2_diff)

        return combine + x1_diff + x2_diff


class up_fusion_inter(nn.Module):
    def __init__(self, out_size, in_size):
        super(up_fusion_inter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_size + in_size, out_size, kernel_size=1, padding=0),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size + in_size, in_size, kernel_size=1, padding=0),
            nn.Conv2d(in_size, in_size, kernel_size=3, padding=1)
        )
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(in_size)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, inputs1, inputs2):
        outputs1 = self.down(inputs1)
        outputs2 = self.up(inputs2)
        res1 = self.conv1(torch.cat([inputs1, outputs2], 1))
        res1 = inputs1 + res1
        res2 = self.conv2(torch.cat([outputs1, inputs2], 1))
        res2 = inputs2 + res2
        return res1, res2


class up_fusion(nn.Module):
    """Upsample the decoder feature and fuse it with the encoder skip feature.

    Args:
        in_size (int): Channels of the feature to be upsampled (``inputs2``).
        out_size (int): Channels of the skip feature (``inputs1``) and of the
            output.
        is_deconv (bool): If True use a transposed convolution for upsampling,
            otherwise a 1x1 convolution followed by bilinear upsampling.
        is_fusion (bool): If True use :class:`fusionBlock` (MDCF) to fuse the
            two features, otherwise a plain 3x3 double convolution.
        kernel_sizes (tuple[int, int]): Passed through to
            :class:`fusionBlock`.  Ignored when ``is_fusion=False``.
    """

    def __init__(self, in_size, out_size, is_deconv, is_fusion=False,
                 kernel_sizes=(5, 7)):
        super(up_fusion, self).__init__()
        self.is_fusion = is_fusion
        if is_fusion:
            self.fs = fusionBlock(dim_in=out_size, kernel_sizes=kernel_sizes)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(out_size * 2, out_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
                nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1),
                nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        if self.is_fusion:
            res = self.fs(inputs1, outputs2)
        else:
            res = self.conv(torch.cat([inputs1, outputs2], 1))
        return res


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class catch_Block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(catch_Block, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.conv1 = nn.Conv2d(dim_in, dim_in * 2, kernel_size=1, padding=0)
        self.conv1_1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in)
        self.conv2 = nn.Conv2d(dim_in * 2, dim_out, kernel_size=1, padding=0)

        self.conv3 = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.size()
        res1 = self.conv1(x)
        res1_1, res1_2 = res1.chunk(2, dim=1)
        res1_1 = self.conv1_1(res1_1)
        res = self.conv2(torch.cat((res1_1, res1_2), dim=1)) + self.conv3(x)
        return res


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 初始卷积核大小 (默认1x1)
            ratio: 初始卷积通道扩张比 (默认生成 out_channels//ratio 个内在特征)
            dw_kernel_size: 幻影卷积核大小 (默认3x3)
        """
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio  # 初始卷积生成的特征通道数

        # 第一阶段：生成内在特征的主卷积
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size,
                      stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        # 第二阶段：生成幻影特征的廉价操作
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, out_channels - init_channels, dw_kernel_size,
                      stride=1, padding=dw_kernel_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(out_channels - init_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)  # [B, init_c, H, W]
        x2 = self.cheap_operation(x1)  # [B, out_c-init_c, H, W]
        return torch.cat([x1, x2], dim=1)  # [B, out_c, H, W]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数不一致，定义1x1卷积层
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 如果输入和输出的通道数不一致，使用1x1卷积调整输入
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out)

        return out


class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2, self).__init__()
        # 第一个 3x3 卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # 避免 inplace 操作

        # 第二个 3x3 卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数不一致，或者步幅不为 1，定义 downsample 层
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入和输出的通道数不一致，或者步幅不为 1，调整输入
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out = out + identity  # 避免 inplace 操作
        out = self.relu(out)

        return out


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMA_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        output = rgb + self.conv4(z)

        return output


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class basic_unet(nn.Module):
    def __init__(self, in_size, out_size, group_size=2, catch=False):
        super(basic_unet, self).__init__()
        self.group_size = group_size
        self.catch = catch

        if self.catch:
            self.conv_low = nn.Sequential(
                nn.Conv2d(in_size // self.group_size, out_size // self.group_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size // self.group_size),
                nn.ReLU(),
                nn.Conv2d(out_size // self.group_size, out_size // self.group_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size // self.group_size),
                nn.ReLU(),
            )

            self.conv_high = nn.Sequential(
                nn.Conv2d(in_size // self.group_size, out_size // self.group_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size // self.group_size),
                nn.ReLU(),
                nn.Conv2d(out_size // self.group_size, out_size // self.group_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size // self.group_size),
                nn.ReLU(),
            )
            self.reconstruct = Reconstruct(out_size)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
                nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )

    def forward(self, x):
        if self.catch:
            x_low, x_high = x.chunk(2, dim=1)
            x_low = self.conv_low(x_low)
            x_high = self.conv_high(x_high)
            res = self.reconstruct(torch.cat((x_low, x_high), dim=1))
        else:
            res = self.conv(x)
        return res


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class lowF_highF(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type='db2'):
        super(lowF_highF, self).__init__()

        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        # self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.wavelet_scale = _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)

    def forward(self, x):
        curr_x = wavelet_transform(x, self.wt_filter)
        shape_x = curr_x.shape
        curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
        curr_x_tag = self.wavelet_scale(curr_x_tag)
        curr_x_tag = curr_x_tag.reshape(shape_x)
        return curr_x_tag[:, :, 0, :, :], curr_x_tag[:, :, 1:4, :, :]


class lowF_highF_IWT(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type='db2'):
        super(lowF_highF_IWT, self).__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

    def forward(self, x_lf, x_hf):
        curr_x = torch.cat([x_lf.unsqueeze(2), x_hf], dim=2)
        iwt_x = inverse_wavelet_transform(curr_x, self.iwt_filter)
        return iwt_x


class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(dim, dim, 3, 1, 1, groups=dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Sequential(
            nn.Conv2d(dim * self.n_levels, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        h, w = x.size()[-2:]
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(x, p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='bilinear', align_corners=False)
            else:
                s = self.mfr[i](x)
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        return out


class RCA(nn.Module):
    def __init__(self, inp, band_kernel_size=7):
        super(RCA, self).__init__()
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=inp),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=inp),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_h = self.conv1(self.pool_h(x))
        x_w = self.conv2(self.pool_w(x))
        x_gather = x_h + x_w
        out = x_gather * x
        return out


class Reconstruct(nn.Module):
    def __init__(self, in_dim):
        super(Reconstruct, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim // 2, in_dim // 2, 3, 1, 1),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 1, 1, 1, 0),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, 1, 0),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)

        xs = x1 - F.adaptive_avg_pool2d(x1, 1)
        x2 = self.conv2(xs)

        fusion = self.fusion(torch.cat((x1, x2), dim=1))

        x1 = x1 + fusion * x1
        x2 = x2 + fusion * x2

        res = self.conv3(torch.cat((x1, x2), dim=1))
        return res


class Block1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ratio = 2

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim, ratio * dim, 1, 1, 0),
            nn.BatchNorm2d(ratio * dim),
            nn.ReLU(),

        )
        self.mid_conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.mid_conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=3 * 2, dilation=3, groups=dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        inputs = x
        x = self.conv_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.mid_conv1(x1)
        x2 = self.mid_conv2(x2)
        fusion = self.conv1(x1 + x2)
        res = self.relu(inputs + fusion)
        return res


@MODELS.register_module()
class UNet_Improve(BaseModule):
    """DADNet backbone: a four-stage U-Net with DFWE and MDCF.

    The network has three down-sampling stages and three corresponding
    up-sampling stages.  ``filters[0]`` is the width of the full-resolution
    stage and ``feature_scale`` divides the canonical widths
    ``(64, 128, 256, 512, 1024)``; the paper uses ``feature_scale=2``, giving
    widths of ``(32, 64, 128, 256)``.

    Args:
        feature_scale (int): Divisor applied to the canonical channel widths.
        is_deconv (bool): Use transposed convolution (True) or 1x1 conv +
            bilinear upsampling (False) in the decoder.
        in_channels (int): Number of input image channels.
        is_batchnorm (bool): Kept for API compatibility with ``UNet``.
        norm_eval (bool): Kept for API compatibility; not used.
        is_catch (bool): Enable **DFWE**.  When True each down-sampling block
            splits its input into two channel groups, processes them with
            separate convolutions and recombines them via
            :class:`Reconstruct`.  Set to False for the ``w/o DFWE`` ablation.
        is_fusion (bool): Enable **MDCF**.  When True the decoder uses
            :class:`fusionBlock` to fuse skip and upsampled features.  Set to
            False for the ``w/o MDCF`` (plain concatenation) ablation.
        kernel_sizes (tuple[int, int]): Kernel sizes of the two depth-wise
            difference branches inside :class:`fusionBlock`.  Defaults to
            ``(5, 7)``; see :class:`fusionBlock` for the ablation variants.
        pool_type (str): Down-sampling pooling, ``'avg'`` (paper default) or
            ``'max'``.

    Note:
        ``UNet_Improve`` is registered under both ``'UNet_Improve'`` and
        ``'DADNet'`` so configs may use either name.
    """

    def __init__(self,
                 feature_scale=4,
                 is_deconv=False,
                 in_channels=3,
                 is_batchnorm=True,
                 norm_eval=False,
                 is_catch=True,
                 is_fusion=True,
                 kernel_sizes=(5, 7),
                 pool_type='avg'):
        super().__init__()
        assert pool_type in ('avg', 'max'), \
            f"pool_type must be 'avg' or 'max', got {pool_type!r}"
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.norm_eval = norm_eval
        self.is_catch = is_catch
        self.is_fusion = is_fusion
        self.kernel_sizes = tuple(kernel_sizes)

        self.filters = [64, 128, 256, 512, 1024]
        self.filters = [int(x / self.feature_scale) for x in self.filters]

        # Each down-sampling block consumes half its input channels when DFWE
        # is enabled (the two halves are processed separately).
        self.group_size = 2 if is_catch else 1
        pool_cls = nn.AvgPool2d if pool_type == 'avg' else nn.MaxPool2d
        self.pool_type = pool_type

        # downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters[0], 3, 1, 1),
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(),
            nn.Conv2d(self.filters[0], self.filters[0], 3, 1, 1),
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(),
        )
        if self.is_catch:
            self.reconstruct1 = Reconstruct(self.filters[0])
        self.pool1 = pool_cls(kernel_size=2)

        self.conv2 = basic_unet(self.filters[0], self.filters[1],
                                catch=self.is_catch, group_size=self.group_size)
        self.pool2 = pool_cls(kernel_size=2)

        self.conv3 = basic_unet(self.filters[1], self.filters[2],
                                catch=self.is_catch, group_size=self.group_size)
        self.pool3 = pool_cls(kernel_size=2)

        self.center = basic_unet(self.filters[2], self.filters[3],
                                 catch=self.is_catch, group_size=self.group_size)

        # upsampling
        self.up_concat3 = up_fusion(self.filters[3], self.filters[2],
                                    self.is_deconv, is_fusion=self.is_fusion,
                                    kernel_sizes=self.kernel_sizes)
        self.up_concat2 = up_fusion(self.filters[2], self.filters[1],
                                    self.is_deconv, is_fusion=self.is_fusion,
                                    kernel_sizes=self.kernel_sizes)
        self.up_concat1 = up_fusion(self.filters[1], self.filters[0],
                                    self.is_deconv, is_fusion=self.is_fusion,
                                    kernel_sizes=self.kernel_sizes)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        if self.is_catch:
            conv1 = self.reconstruct1(conv1)

        conv2 = self.conv2(self.pool1(conv1))
        conv3 = self.conv3(self.pool2(conv2))
        center = self.center(self.pool3(conv3))

        up3 = self.up_concat3(conv3, center)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        return [up1]


# ``DADNet`` is the name used in the paper; both keys resolve to the same class.
DADNet = UNet_Improve
MODELS.register_module(name='DADNet', module=UNet_Improve)
