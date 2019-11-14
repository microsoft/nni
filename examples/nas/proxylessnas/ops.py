# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from collections import OrderedDict
import torch.nn as nn


OPS = {
    'Identity': lambda in_C, out_C, stride: IdentityLayer(in_C, out_C, ops_order='weight_bn_act'),
    'Zero': lambda in_C, out_C, stride: ZeroLayer(stride=stride),
    '3x3_MBConv1': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 1),
    '3x3_MBConv2': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 2),
    '3x3_MBConv3': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 3),
    '3x3_MBConv4': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 4),
    '3x3_MBConv5': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 5),
    '3x3_MBConv6': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 6),
    '5x5_MBConv1': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 1),
    '5x5_MBConv2': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 2),
    '5x5_MBConv3': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 3),
    '5x5_MBConv4': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 4),
    '5x5_MBConv5': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 5),
    '5x5_MBConv6': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 6),
    '7x7_MBConv1': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 1),
    '7x7_MBConv2': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 2),
    '7x7_MBConv3': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 3),
    '7x7_MBConv4': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 4),
    '7x7_MBConv5': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 5),
    '7x7_MBConv6': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 6)
}

#========================================

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)

#========================================

class MyModule(nn.Module):
    
    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

class MobileInvertedResidualBlock(MyModule):
    
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)

#========================================

def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
                out_h * out_w / layer.groups
    return delta_ops

class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # noinspection PyUnresolvedReferences
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

class MyModule(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

class My2DLayer(MyModule):
    
    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)


class DepthConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['depth_conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.in_channels, bias=False
        )
        weight_dict['point_conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        x = self.depth_conv(x)
        point_flop = count_conv_flop(self.point_conv, x)
        x = self.point_conv(x)
        return depth_flop + point_flop, self.forward(x)


class PoolingLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class IdentityLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class LinearLayer(MyModule):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_MBConv%d' % (self.kernel_size, self.kernel_size, self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)

        flop3 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3, x

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(MyModule):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True
