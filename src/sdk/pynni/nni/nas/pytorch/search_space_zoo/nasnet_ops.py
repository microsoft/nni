import torch
import torch.nn as nn
import torch.nn.functional as F


class Pool(nn.Module):
    def __init__(self, pool_type, in_channels, out_channels, kernel_size, stride, padding,
                 affine=True, momentum=0.1, eps=0.001):
        super(Pool, self).__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError('Key pool_type in POol can only be \'max\' or \'avg\'')
        self.conv, self.bn = None, None
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
            self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        x = self.pool(x)
        if self.conv and self.bn:
            x = self.conv(x)
            x = self.bn(x)
        return x


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride, affine=True, momentum=0.1, eps=0.001):
        super(Identity, self).__init__()
        if in_channels == out_channels and stride == 1:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine)
            )

    def forward(self, x):
        return self.op(x)


class StdConv(nn.Module):
    def __init__(self, in_channels, out_channels, affine=True, momentum=0.1, eps=0.001):
        super(StdConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        return self.bn(x)
    
    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(Conv, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        return self.pointwise_conv2d(x)
    

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, stem=False, bias=False,
                 affine=True, momentum=0.1, eps=0.001):
        super(SeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels if stem else in_channels
        self.operation = nn.Sequential(
            nn.ReLU(),
            Conv(in_channels, mid_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(mid_channels, eps=eps, momentum=momentum, affine=affine),
            Conv(mid_channels, out_channels, kernel_size, 1, padding, bias=bias),
            nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine)
        )

    def forward(self, x):
        return self.operation(x)

