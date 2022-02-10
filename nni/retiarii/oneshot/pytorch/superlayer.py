
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from re import S
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoice

class ToSample:
    """
    Base class for all things to be sampled.

    Parameters
    ----------
    sampled : int
        the index of the sampled candidate
    """
    def __init__(self, label, candidates = [], sampled = 0) -> None:
        self.label = label
        self.candidates = candidates
        self.sampled = sampled
    
    @property
    def sampled_candidate(self):
        return self.candidates[self.sampled]
    
    def __len__(self):
        return len(self.candidates)


class ENASValueChoice(ToSample):
    """
    """
    def __init__(self, value_choice):
        super().__init__(value_choice.label, value_choice.candidates)
        self.n_chosen = 1
    # ENAS 与 random 共用
    # Conv2D valueChoice

class RandomValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, value_choice.candidates)


def sampled_candidate(attr):
    if isinstance(attr, ToSample):
        attr = attr.sampled_candidate
    return attr

def max_candidate(attr):
    if isinstance(attr, ToSample):
        attr = max(attr.candidates)
    return attr

# SuperLinear(nn.Linear) 放一个单独文件
class PathSamplingSuperLinear(nn.Linear):
    def __init__(self, module) -> None:
        args = module.trace_kwargs
        self._in_features = args['in_features']
        max_in_features = max_candidate(self._in_features)
        
        self._out_features = args['out_features']
        max_out_features = max_candidate(self._out_features)

        bias = args.get('bias', True)
        device = args.get('device', None)
        dtype = args.get('dtype', None)
        
        super().__init__(max_in_features, max_out_features, bias, device, dtype)

    def forward(self, x):
        # 如果是 valuechoice 就去读 sample 的值，否则就是固定值
        in_dim = sampled_candidate(self._in_features)
        out_dim = sampled_candidate(self._out_features)
        
        weights = self.weight[:out_dim, :in_dim]
        bias = self.bias[:out_dim]
        
        return F.linear(x, weights, bias)

class PathSamplingSuperConv2d(nn.Conv2d):
    # 暂时只支持正方形的 kernel
    # 暂不支持 group
    # 也不支持嵌套
    def __init__(self, module):
        args = module.trace_kwargs
        # out_channel 组卷积核，每组 in_channel 个，每个 kernel_size * kernel_size 这么大。
        self._in_channels = args['in_channels']
        max_in_channel = max_candidate(self._in_channels)
        self._out_channels = args['out_channels']
        max_out_channel = max_candidate(self._out_channels)
        self._kernel_size = args['kernel_size']
        max_kernel_size = max_candidate(self._kernel_size)
        self._stride = args.get('stride', 1)
        self._padding = args.get('padding', 0)
        self._dilation = args.get('dilation', 1)
        self._groups = args.get('groups', 1)
        self._bias = args.get('bias', False)
        _padding_mode = args.get('padding_mode', 'zeros')
        _device = args.get('device', None)
        _dtype = args.get('dtype', None)
        super().__init__(max_in_channel, max_out_channel, max_kernel_size, padding_mode = _padding_mode, device = _device, dtype = _dtype)
    
    def forward(self, input):
        in_chn = sampled_candidate(self._in_channels)
        out_chn = sampled_candidate(self._out_channels)
        kernel_size = sampled_candidate(self._kernel_size)

        # conv 和 linear 不一样，前面的 stride 和 padding 是可能会影响到后面层的 size 的。
        # 这就要求对 value choice 的 sample时，要把 sampled index 和 candiate 分开了。。。。
        # 然后用户要自行保证前后的 size 对准
        self.stride = sampled_candidate(self._stride) # tuple 的情况支持一下
        self.padding = sampled_candidate(self._padding)
        self.dilation = sampled_candidate(self._dilation)
        # 支持 group 的话，我应该是对第二维进行拆分。
        self.groups = sampled_candidate(self._groups)

        # 暂且只支持正方形的 kernel
        # weight.shape = [out_chn, in_chn, kernel_size]
        # 如何取部分 kernel? 是从中央取
        weight = self.weight[:out_chn, :in_chn, :kernel_size, :kernel_size]
        bias = self.bias[:out_chn]
        return self._conv_forward(input, weight, bias)

    