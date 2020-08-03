import torch
import torch.nn as nn


class ReLUConvBN(nn.Module):
    """
    Parameters
    ---
    configs:
        # TODO: remove this arg
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    padding: int
        zero-padding added to both sides of the input
    dilation: int
        spacing between kernel elements
    """
    def __init__(self, configs, C_in, C_out, kernel_size, stride, padding, dilation):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=configs.bn_affine, momentum=configs.bn_momentum,
                           track_running_stats=configs.bn_track_running_stats)
        )

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        return self.op(x)


class Pooling(nn.Module):
    """
    Parameters
    ---
    configs:
        # TODO: remove this arg
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    mode: str
        must be ``avg`` or ``max``
    """
    def __init__(self, configs, C_in, C_out, stride, mode):
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(configs, C_in, C_out, 1, 1, 0)
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in Pooling".format(mode))

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        if self.preprocess:
            x = self.preprocess(x)
        return self.op(x)


class Zero(nn.Module):
    """
    Parameters
    ---
    configs:
        # TODO: remove this arg
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    """
    def __init__(self, configs, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class FactorizedReduce(nn.Module):
    def __init__(self, configs, C_in, C_out, stride):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=configs.bn_affine, momentum=configs.bn_momentum,
                                 track_running_stats=configs.bn_track_running_stats)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ResNetBasicblock(nn.Module):

    def __init__(self, configs, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(configs, inplanes, planes, 3, stride, 1, 1)
        self.conv_b = ReLUConvBN(configs, planes, planes, 3, 1, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(configs, inplanes, planes, 1, 1, 0, 1)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)  # residual
        return inputs + basicblock