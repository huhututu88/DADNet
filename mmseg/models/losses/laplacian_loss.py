# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmseg.registry import MODELS
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as fnn


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :]).cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)
    pyr.append(current)
    return pyr


def get_laplacian_loss_whole_img(predict, alpha):
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    gauss_kernel = build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=True)
    pyr_alpha = laplacian_pyramid(alpha_f, gauss_kernel, 5)
    pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
    laplacian_loss = sum(fnn.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))
    return laplacian_loss


@MODELS.register_module()
class LaplacianLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 loss_name='loss_laplacian'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        cls_score_defect = cls_score[:, 1:, :, :].sum(dim=1)
        label_defect = label[label[:, :, :] != 0] = 1
        loss_laplacian = get_laplacian_loss_whole_img(cls_score_defect, label_defect)
        return loss_laplacian

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
