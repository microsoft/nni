# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Union, Tuple, Callable, Optional
from collections.abc import Iterable
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .target_space import TargetType


## ======= some process funcs for conv copied from torch ======
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

def reversed_padding_repeated_twice(conv_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]):
    if isinstance(conv_module.padding, str):
        _reversed_padding_repeated_twice = [0, 0] * len(conv_module.kernel_size)
        if conv_module.padding == 'same':
            for d, k, i in zip(conv_module.dilation, conv_module.kernel_size,
                                range(len(conv_module.kernel_size) - 1, -1, -1)):
                total_padding = d * (k - 1)
                left_pad = total_padding // 2
                _reversed_padding_repeated_twice[2 * i] = left_pad
                _reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
    else:
        _reversed_padding_repeated_twice = _reverse_repeat_tuple(conv_module.padding, 2)

    return _reversed_padding_repeated_twice

def conv_forward(inputs: torch.Tensor, conv_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], \
                 weight: torch.Tensor, bias: Optional[torch.Tensor]) -> Tensor:
    assert isinstance(conv_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)), \
        f"expected module type in [Conv1d, Conv2d, Conv3d], but got {type(conv_module)}"

    def conv_forward_func(conv_func, padding_func):
        # copied from torch
        if conv_module.padding_mode != 'zeros':
            return conv_func(F.pad(inputs, reversed_padding_repeated_twice(conv_module), mode=conv_module.padding_mode),
                            weight, bias, conv_module.stride,
                            padding_func(0), conv_module.dilation, conv_module.groups)
        return conv_func(inputs, weight, bias, conv_module.stride,
                        conv_module.padding, conv_module.dilation, conv_module.groups)

    if type(conv_module) == torch.nn.Conv1d:
        return conv_forward_func(F.conv1d, _single)
    elif type(conv_module) == torch.nn.Conv2d:
        return conv_forward_func(F.conv2d, _pair)
    elif type(conv_module) == torch.nn.Conv3d:
        return conv_forward_func(F.conv3d, _triple)
    else:
        raise TypeError(f"Only support modules in [conv1d, conv2d, conv3d], but got {type(conv_module)}")


def get_bias(wrapper, params_dict):
    if 'bias' in params_dict:
        assert params_dict['bias'] is not None
        return params_dict['bias']
    elif wrapper.is_bias == 'Tensor':
        assert getattr(wrapper, "is_register_bias", False)
        return wrapper.bias
    else:
        return None


def set_bias(wrapper, fused_bias):
    return fused_bias if wrapper.is_bias == "Tensor" else torch.nn.Parameter(fused_bias, False)


def fuse_modules(wrapper, params_dict, *args, **kwargs):
    fused_modules = wrapper.fused_modules
    types = tuple(type(module) for module in fused_modules)

    fuse_method = _DEFAULT_OP_LIST_TO_FUSER_METHOD.get(types, None)
    if fuse_method is None:
        raise TypeError(f"{types} is not supported for model fusion, please register it \
                                  in the _DEFAULT_OP_LIST_TO_FUSER_METHOD")

    return fuse_method(wrapper, fused_modules, params_dict, *args, **kwargs)

# ============ fuse method =============
def fuse_conv_bn(wrapper, fused_modules, params_dict, *args, **kwargs):
    q_param_dict = {k: v.target for k, v in wrapper.quantization_target_spaces.items() if v.type is TargetType.PARAMETER}
    params_dict = params_dict.copy()
    bn_module = fused_modules[1]
    conv_module = wrapper.module

    assert bn_module.num_features == conv_module.out_channels, \
        'The output channel of Conv must match num_features of BatchNorm'

    if 'weight' not in q_param_dict:
        raise ValueError(f"In the fusion process of conv_bn, the weight of {wrapper.name} needs to be quantized")

    with torch.no_grad():
        #compute bias
        bias: Union[torch.Tensor, None] = get_bias(wrapper, params_dict)
        output = conv_forward(*args, **kwargs, conv_module=conv_module, \
            weight=q_param_dict['weight'], bias=bias)
        #statistics mean and var when track_running_stats = False
        if not bn_module.track_running_stats:
            sta_output = output.detach().clone()
            sta_output = torch.transpose(sta_output, 0, 1)
            sta_output = sta_output.contiguous().view(conv_module.out_channels, -1)
            mean = sta_output.mean(1).detach()
            var = sta_output.var(1).detach()
            bn_rm = mean
            bn_var = var
        # if track == True
        else:
            _ = bn_module._nni_wrapper.module_forward(output)
            bn_rm = bn_module.running_mean
            bn_var = bn_module.running_var

    bn_eps = bn_module.eps
    # case: affine
    bn_w = bn_module.weight if bn_module.affine else torch.ones_like(bn_rm)
    bn_b = bn_module.bias if bn_module.affine else torch.zeros_like(bn_rm)
    # conv weight and bias
    conv_weight = q_param_dict["weight"]
    conv_bias = get_bias(wrapper, params_dict)
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_rm)
    # fuse weight and bias
    shape = [-1, 1] + [1] * (len(conv_weight.shape) - 2)
    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)
    fused_conv_w = conv_weight * (bn_w * bn_var_rsqrt).reshape(shape)
    fused_conv_b = (conv_bias - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    params_dict['weight'] = fused_conv_w
    params_dict['bias'] = set_bias(wrapper, fused_conv_b)

    return params_dict, []


def fuse_linear_bn(wrapper, fused_modules, params_dict, *args, **kwargs):
    q_param_dict = {k: v.target for k, v in wrapper.quantization_target_spaces.items() if v.type is TargetType.PARAMETER}
    params_dict = params_dict.copy()
    bn_module = fused_modules[1]
    linear_module = wrapper.module

    if 'weight' not in q_param_dict:
        raise ValueError(f"In the fusion process of linear_bn, the weight of {wrapper.name} needs to be quantized")

    assert bn_module.num_features == linear_module.out_features, \
        'the output features of Linear must match num_features of BatchNorm'

    with torch.no_grad():
        bias = get_bias(wrapper, params_dict)
        output = F.linear(*args, **kwargs, weight=q_param_dict['weight'], bias=bias)
        # statistic mean and var
        if not bn_module.track_running_stats:
            sta_output = output.detach().clone()
            sta_output = sta_output.contiguous().view(-1, linear_module.out_features)
            mean = sta_output.mean(0).detach()
            var = sta_output.var(0).detach()
            bn_rm = mean
            bn_var = var
        else:
            _ = bn_module._nni_wrapper.module_forward(output)
            bn_rm = bn_module.running_mean
            bn_var = bn_module.running_var

    bn_eps = bn_module.eps
    # case: affine
    bn_w = bn_module.weight if bn_module.affine else torch.ones_like(bn_rm)
    bn_b = bn_module.bias if bn_module.affine else torch.zeros_like(bn_rm)
    # linear weight
    l_weight = q_param_dict["weight"]
    l_bias = get_bias(wrapper, params_dict)
    if l_bias is None:
        l_bias = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_var + bn_eps)

    fused_lin_w = l_weight * bn_scale.unsqueeze(-1)
    fused_lin_b = (l_bias - bn_rm) * bn_scale + bn_b

    params_dict['weight'] = fused_lin_w
    params_dict['bias'] = set_bias(wrapper, fused_lin_b)

    return params_dict, []


def fuse_linear_bn_relu(wrapper, fused_modules, params_dict, *args, **kwargs):
    assert len(fused_modules) == 3 and isinstance(fused_modules[2], torch.nn.ReLU)
    # firstly, fuse linear and bn
    params_dict, _ = fuse_linear_bn(wrapper, fused_modules, params_dict, *args, **kwargs)

    return params_dict, [fused_modules[2]]


def fuse_conv_bn_relu(wrapper, fused_modules, params_dict, *args, **kwargs):
    assert len(fused_modules) == 3 and isinstance(fused_modules[2], torch.nn.ReLU)
    # firstly, fuse conv and bn
    params_dict, _ = fuse_conv_bn(wrapper, fused_modules, params_dict, *args, **kwargs)

    return params_dict, [fused_modules[2]]


def fuse_conv_relu(wrapper, fused_modules, params_dict, *args, **kwargs):
    assert type(fused_modules[1]) == nn.ReLU

    return params_dict, [fused_modules[1]]


def fuse_linear_relu(wrapper, fused_modules, params_dict, *args, **kwargs):
    assert type(fused_modules[1]) == nn.ReLU

    return params_dict, [fused_modules[1]]


def fuse_bn_relu(wrapper, fused_modules, params_dict, *args, **kwargs):
    assert type(fused_modules[1]) == nn.ReLU

    return params_dict, [fused_modules[1]]


_DEFAULT_OP_LIST_TO_FUSER_METHOD: Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): fuse_conv_relu,
    (nn.Conv2d, nn.ReLU): fuse_conv_relu,
    (nn.Conv3d, nn.ReLU): fuse_conv_relu,
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nn.BatchNorm1d, nn.ReLU): fuse_linear_bn_relu,
    (nn.Linear, nn.ReLU): fuse_linear_relu,
    (nn.BatchNorm1d, nn.ReLU): fuse_bn_relu,
    (nn.BatchNorm2d, nn.ReLU): fuse_bn_relu,
    (nn.BatchNorm3d, nn.ReLU): fuse_bn_relu,
}
