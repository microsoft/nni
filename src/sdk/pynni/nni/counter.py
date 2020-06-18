# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from nni.compression.torch.compressor import PrunerModuleWrapper
import logging

_logger = logging.getLogger(__name__)

try:
    from thop import profile
except ImportError:
    _logger.warning('Please install thop first.')

def count_flops_params(model: nn.Module, input_size=None, verbose=False):
    """
    Count FLOPs and Params of the given model. This function would identify the mask on the module and take the pruend shape into consideration. Note that, we only identify the output shape of the module, so the calculated FLOPs will be larger than real number.

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
    for idx, m in enumerate(model.modules()):
        weight_mask = None
        m_type = type(m)
        if m_type in custom_ops:
            if isinstance(prev_m, PrunerModuleWrapper):
                weight_mask = prev_m.weight_mask

            m.register_buffer('weight_mask', weight_mask)
            hook_module_list.append(m)
        prev_m = m

    flops, params = profile(model, inputs=(inputs, ), custom_ops=custom_ops, verbose=False)

    for m in hook_module_list:
        m._buffers.pop("weight_mask")

    return flops, params

def count_convNd_mask(m, x, y):
    output_channel = y.size()[1]
    output_size =  torch.zeros(y.size()[2:]).numel()
    kernel_size = torch.zeros(m.weight.size()[2:]).numel()

    bias_flops = 1 if m.bias is not None else 0

    if m.weight_mask is not None:
        output_channel = m.weight_mask.sum() // (m.in_channels * kernel_size)

    total_ops = output_channel * output_size * (m.in_channels // m.groups * kernel_size + bias_flops)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_linear_mask(m, x, y):
    output_channel = y.size()[1]
    output_size =  torch.zeros(y.size()[2:]).numel()

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
