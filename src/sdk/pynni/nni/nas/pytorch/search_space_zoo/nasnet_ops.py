import torch
import torch.nn as nn
import torch.nn.functional as F


class Pool(nn.Module):
    def __init__(self, pool_type, in_channels, out_channels, kernel_size, stride, padding,
                 affine=True, momentum=0.1, eps=0.001):
        super(Pool, self).__init__()
        self.conv = StdConv(in_channels, out_channels)
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError('Key pool_type in POol can only be \'max\' or \'avg\'')
        self.bn = nn.BatchNorm2d(out_channels, affine=affine, momentum=momentum, eps=eps)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.bn(x)


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride, affine=True, momentum=0.1, eps=0.001):
        super(Identity, self).__init__()
        if in_channels == out_channels and stride == 1:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                StdConv(in_channels, out_channels),
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(Conv, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size,
                                          stride=stride,
                                          padding=padding,
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
        self.op = nn.Sequential(
            nn.ReLU(),
            Conv(in_channels, mid_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(mid_channels, eps=eps, momentum=momentum, affine=affine),
            Conv(mid_channels, out_channels, kernel_size, 1, padding, bias=bias),
            nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
