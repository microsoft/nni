# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from nni.compression.torch.compressor import PrunerModuleWrapper

try:
    from thop import profile
except Exception as e:
    print('thop is not found, please install the python package: thop')
    raise


def count_flops_params(model: nn.Module, input_size, verbose=True):
    """
    Count FLOPs and Params of the given model.
    This function would identify the mask on the module
    and take the pruned shape into consideration.
    Note that, for sturctured pruning, we only identify
    the remained filters according to its mask, which
    not taking the pruned input channels into consideration,
    so the calculated FLOPs will be larger than real number.

    Parameters
    ---------
    model : nn.Module
        target model.
    input_size: list, tuple
        the input shape of data


    Returns
    -------
    flops: float
        total flops of the model
    params:
        total params of the model
    """

    assert input_size is not None

    device = next(model.parameters()).device
    inputs = torch.randn(input_size).to(device)

    hook_module_list = []
    prev_m = None
    for m in model.modules():
        weight_mask = None
        m_type = type(m)
        if m_type in custom_ops:
            if isinstance(prev_m, PrunerModuleWrapper):
                weight_mask = prev_m.weight_mask

            m.register_buffer('weight_mask', weight_mask)
            hook_module_list.append(m)
        prev_m = m

    flops, params = profile(model, inputs=(inputs, ), custom_ops=custom_ops, verbose=verbose)


    for m in hook_module_list:
        m._buffers.pop("weight_mask")
    # Remove registerd buffer on the model, and fixed following issue:
    # https://github.com/Lyken17/pytorch-OpCounter/issues/96
    for m in model.modules():
        if 'total_ops' in m._buffers:
            m._buffers.pop("total_ops")
        if 'total_params' in m._buffers:
            m._buffers.pop("total_params")

    return flops, params

def count_convNd_mask(m, x, y):
    """
    The forward hook to count FLOPs and Parameters of convolution operation.

    Parameters
    ----------
    m : torch.nn.Module
        convolution module to calculate the FLOPs and Parameters
    x : torch.Tensor
        input data
    y : torch.Tensor
        output data
    """
    output_channel = y.size()[1]
    output_size = torch.zeros(y.size()[2:]).numel()
    kernel_size = torch.zeros(m.weight.size()[2:]).numel()

    bias_flops = 1 if m.bias is not None else 0

    if m.weight_mask is not None:
        output_channel = m.weight_mask.sum() // (m.in_channels * kernel_size)

    total_ops = output_channel * output_size * (m.in_channels // m.groups * kernel_size + bias_flops)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_linear_mask(m, x, y):
    """
    The forward hook to count FLOPs and Parameters of linear transformation.

    Parameters
    ----------
    m : torch.nn.Module
        linear to calculate the FLOPs and Parameters
    x : torch.Tensor
        input data
    y : torch.Tensor
        output data
    """
    output_channel = y.size()[1]
    output_size = torch.zeros(y.size()[2:]).numel()

    bias_flops = 1 if m.bias is not None else 0

    if m.weight_mask is not None:
        output_channel = m.weight_mask.sum() // m.in_features

    total_ops = output_channel * output_size * (m.in_features + bias_flops)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


custom_ops = {
    nn.Conv1d: count_convNd_mask,
    nn.Conv2d: count_convNd_mask,
    nn.Conv3d: count_convNd_mask,
    nn.Linear: count_linear_mask,
}
