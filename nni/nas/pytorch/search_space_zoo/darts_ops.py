# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, p=0.):
        """
        Drop path with probability.

        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), 1, 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool with BN. ``pool_type`` must be ``max`` or ``avg``.

    Parameters
    ---
    pool_type: str
        choose operation
    C: int
        number of channels
    kernal_size: int
	    size of the convolving kernel
    stride: int
	    stride of the convolution
    padding: int
	    zero-padding added to both sides of the input
    affine: bool
        is using affine in BatchNorm
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Sequential):
    """
    Standard conv: ReLU - Conv - BN

    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    kernel_size: int
        size of the convolution kernel
    padding:
        zero-padding added to both sides of the input
    affine: bool
        is using affine in BatchNorm
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential
        for idx, ops in enumerate((nn.ReLU(), nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
                                   nn.BatchNorm2d(C_out, affine=affine))):
            self.add_module(str(idx), ops)


class FacConv(nn.Module):
    """
    Factorized conv: ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.

    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    kernal_size:
        size of the convolving kernel
    padding:
        zero-padding added to both sides of the input
    dilation: int
        spacing between kernel elements.
    affine: bool
        is using affine in BatchNorm
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.

    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    kernal_size:
        size of the convolving kernel
    padding:
        zero-padding added to both sides of the input
    dilation: int
        spacing between kernel elements.
    affine: bool
        is using affine in BatchNorm
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise (stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
