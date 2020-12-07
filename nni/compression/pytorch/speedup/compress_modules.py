# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .infer_shape import ModuleMasks

_logger = logging.getLogger(__name__)

replace_module = {
    'BatchNorm2d': lambda module, mask: replace_batchnorm2d(module, mask),
    'Conv2d': lambda module, mask: replace_conv2d(module, mask),
    'MaxPool2d': lambda module, mask: no_replace(module, mask),
    'AvgPool2d': lambda module, mask: no_replace(module, mask),
    'AdaptiveAvgPool2d': lambda module, mask: no_replace(module, mask),
    'ReLU': lambda module, mask: no_replace(module, mask),
    'ReLU6': lambda module, mask: no_replace(module, mask),
    'Sigmoid': lambda module, mask: no_replace(module, mask),
    'Linear': lambda module, mask: replace_linear(module, mask),
    'Dropout': lambda module, mask: no_replace(module, mask),
    'Dropout2d': lambda module, mask: no_replace(module, mask),
    'Dropout3d': lambda module, mask: no_replace(module, mask)
}

def no_replace(module, mask):
    """
    No need to replace
    """
    _logger.debug("no need to replace")
    return module

def replace_linear(linear, mask):
    """
    Parameters
    ----------
    linear : torch.nn.Linear
        The linear module to be replace
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.Linear
        The new linear module
    """
    assert isinstance(mask, ModuleMasks)
    assert mask.input_mask is not None
    assert mask.output_mask is None
    assert not mask.param_masks
    index = mask.input_mask.mask_index[-1]
    in_features = index.size()[0]
    _logger.debug("replace linear with new in_features: %d", in_features)
    new_linear = torch.nn.Linear(in_features=in_features,
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    new_linear.to(linear.weight.device)
    new_linear.weight.data = torch.index_select(linear.weight.data, -1, index.to(linear.weight.device))
    if linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.data)
    return new_linear

def replace_batchnorm2d(norm, mask):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm2d
        The batchnorm module to be replace
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.BatchNorm2d
        The new batchnorm module
    """
    assert isinstance(mask, ModuleMasks)
    assert 'weight' in mask.param_masks and 'bias' in mask.param_masks
    index = mask.param_masks['weight'].mask_index[0]
    num_features = index.size()[0]
    _logger.debug("replace batchnorm2d with num_features: %d", num_features)
    new_norm = torch.nn.BatchNorm2d(num_features=num_features,
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, index)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, index)
    if norm.track_running_stats:
        new_norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, index)
        new_norm.running_var.data = torch.index_select(norm.running_var.data, 0, index)
    return new_norm

def replace_conv2d(conv, mask):
    """
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The conv2d module to be replaced
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    assert isinstance(mask, ModuleMasks)
    if mask.input_mask is None:
        in_channels = conv.in_channels
    else:
        in_channels_index = mask.input_mask.mask_index[1]
        in_channels = in_channels_index.size()[0]
    if mask.output_mask is None:
        out_channels = conv.out_channels
    else:
        out_channels_index = mask.output_mask.mask_index[1]
        out_channels = out_channels_index.size()[0]
    groups = conv.groups
    if conv.in_channels == conv.out_channels == conv.groups:
        # remove groups for depthwise layers
        assert in_channels == out_channels
        groups = in_channels
    _logger.debug("replace conv2d %s with in_channels: %d, out_channels: %d", mask.module_name, in_channels, out_channels)
    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=groups,
                               bias=conv.bias is not None,
                               padding_mode=conv.padding_mode)

    new_conv.to(conv.weight.device)
    tmp_weight_data = tmp_bias_data = None

    if mask.output_mask is not None:
        tmp_weight_data = torch.index_select(conv.weight.data, 0, out_channels_index)
        if conv.bias is not None:
            tmp_bias_data = torch.index_select(conv.bias.data, 0, out_channels_index)
    else:
        tmp_weight_data = conv.weight.data
    # For the convolutional layers that have more than one group
    # we need to copy the weight group by group, because the input
    # channal is also divided into serveral groups and each group
    # filter may have different input channel indexes.
    input_step = int(conv.in_channels / conv.groups)
    in_channels_group = int(in_channels / groups)
    filter_step = int(out_channels / groups)
    if mask.input_mask is not None and not (in_channels == out_channels == groups):
        for groupid in range(conv.groups):
            start = groupid * input_step
            end = (groupid + 1) * input_step
            current_input_index = list(filter(lambda x: start <= x and x < end, in_channels_index.tolist()))
            if not current_input_index:
                # there is no kept channel in current group
                continue
            # shift the global index into the group index
            current_input_index = [x-start for x in current_input_index]
            # if the groups is larger than 1, the input channels of each
            # group should be pruned evenly.
            assert len(current_input_index) == in_channels_group, \
                'Input channels of each group are not pruned evenly'
            current_input_index = torch.tensor(current_input_index).to(tmp_weight_data.device) # pylint: disable=not-callable
            f_start = groupid * filter_step
            f_end = (groupid + 1) * filter_step
            new_conv.weight.data[f_start:f_end] = torch.index_select(tmp_weight_data[f_start:f_end], 1, current_input_index)
    else:
        new_conv.weight.data.copy_(tmp_weight_data)

    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data if tmp_bias_data is None else tmp_bias_data)

    return new_conv
