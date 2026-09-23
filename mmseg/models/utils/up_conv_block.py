# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer


class residual_catch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_catch, self).__init__()
        group = 7
        dilate = 3
        self.scale = 4
        self.conv_layers_level1 = nn.Sequential(
            nn.Conv2d(in_channels // self.scale,
                      in_channels // self.scale,
                      kernel_size=(1, group),
                      padding='same'),
            nn.BatchNorm2d(in_channels // self.scale),
            nn.ReLU(),
            nn.Conv2d(in_channels // self.scale,
                      in_channels // self.scale,
                      kernel_size=1,
                      padding='same'),
        )
        self.conv_layers_level2 = nn.Sequential(
            nn.Conv2d(in_channels // self.scale,
                      in_channels // self.scale,
                      kernel_size=(group, 1),
                      padding='same'),
            nn.BatchNorm2d(in_channels // self.scale),
            nn.ReLU(),
            nn.Conv2d(in_channels // self.scale,
                      in_channels // self.scale,
                      kernel_size=1,
                      padding='same'),
        )
        self.conv_layers_level3 = nn.Sequential(
            nn.Conv2d(
                in_channels // self.scale,
                in_channels // self.scale,
                kernel_size=3,
                padding=dilate,
                dilation=dilate),
            nn.BatchNorm2d(in_channels // self.scale),
            nn.ReLU(),
            nn.Conv2d(in_channels // self.scale,
                      in_channels // self.scale,
                      kernel_size=1,
                      padding='same'),
        )

    def forward(self, x):
        temp = x + self.conv_layers_level1(x) + self.conv_layers_level2(x)
        temp = temp + self.conv_layers_level3(temp)
        res = self.conv1(temp)
        return res


class fusionSD(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusionSD, self).__init__()
        self.conv_avg1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
        )
        self.conv_avg2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
        )

        self.conv_max1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
        )
        self.conv_max2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
        )

        self.conv_fusion = nn.Conv2d(2, 1, kernel_size=7, padding='same')

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding='same')
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in + ch_out, ch_out, kernel_size=3, padding='same'),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)

        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)

        avg_out_sim = avg_out1 + avg_out2
        avg_out_diff = avg_out1 - avg_out2
        max_out_sim = max_out1 + max_out2
        max_out_diff = max_out1 - max_out2

        avg_out = self.conv_avg1(avg_out_sim) + self.conv_avg2(avg_out_diff)
        max_out = self.conv_max1(max_out_sim) + self.conv_max2(max_out_diff)

        supply = self.conv_fusion(torch.cat([avg_out, max_out], dim=1)).sigmoid() * x1 + self.conv1(x1).sigmoid() * x1
        res = self.conv2(torch.cat((supply, x2), dim=1))
        return res


class UpConvBlock_Improve(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.fs1 = fusionSD(ch_in=skip_channels, ch_out=skip_channels)
        # self.fs2 = fusionSD(ch_in=skip_channels, ch_out=skip_channels)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        # out = torch.cat([skip, x], dim=1)
        x_refine = self.fs1(x, skip)
        # skip_refine = self.fs2(skip, x)
        # out = torch.cat([skip_refine, x_refine], dim=1)
        # out = self.conv_block(x_refine)

        return x_refine


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out
