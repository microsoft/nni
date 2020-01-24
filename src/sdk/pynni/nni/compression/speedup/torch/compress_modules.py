# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .infer_shape import CoarseMask, ModuleMasks

replace_module = {
    'BatchNorm2d': lambda module, mask: replace_batchnorm2d(module, mask),
    'Conv2d': lambda module, mask: replace_conv2d(module, mask),
    'MaxPool2d': lambda module, mask: no_replace(module, mask),
    'ReLU': lambda module, mask: no_replace(module, mask),
    'Linear': lambda module, mask: replace_linear(module, mask)
}

def no_replace(module, mask):
    """
    """
    return module

def replace_linear(linear, mask):
    """
    """
    assert isinstance(mask, ModuleMasks)
    assert mask.input_mask is not None
    assert mask.output_mask is None
    assert not mask.param_masks
    index = mask.input_mask.mask_index[-1]
    print(mask.input_mask.mask_index)
    in_features = index.size()[0]
    print('linear: ', in_features)
    new_linear = torch.nn.Linear(in_features=in_features,
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    print(linear.weight.data.size())
    print(new_linear.weight.data.size())
    print(linear.weight.t().size())
    print(new_linear.weight.t().size())
    new_linear.weight.data = torch.index_select(linear.weight.data, -1, index.to('cuda:0'))
    print(new_linear.weight.data.size())
    if linear.bias is not None:
        print(linear.bias.data.size())
        new_linear.bias.data = torch.index_select(linear.bias.data, 0, index.to('cuda:0'))
        print(new_linear.bias.data.size())
    print("last print: ", new_linear.weight.t().size())
    return new_linear

def replace_batchnorm2d(norm, mask):
    """
    """
    assert isinstance(mask, ModuleMasks)
    assert 'weight' in mask.param_masks and 'bias' in mask.param_masks
    index = mask.param_masks['weight'].mask_index[0]
    num_features = index.size()[0]
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
    """
    # fine-grained tensor sparse
    #...
    # coarse-grained shape sparse
    #...
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
    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=1, # currently only support groups is 1
                               bias=conv.bias,
                               padding_mode=conv.padding_mode)
    #print('weight: ', conv.weight.get_device())
    #print('bias', conv.bias.get_device())
    tmp_weight_data = tmp_bias_data = None
    if mask.output_mask is not None:
        tmp_weight_data = torch.index_select(conv.weight.data, 0, out_channels_index)
        if conv.bias is not None:
            print('bias is not None')
            tmp_bias_data = torch.index_select(conv.bias.data, 0, out_channels_index)
    # NOTE: does not support group
    if mask.input_mask is not None:
        tmp_weight_data = torch.index_select(tmp_weight_data, 1, in_channels_index)
    if tmp_weight_data is not None:
        new_conv.weight.data = tmp_weight_data
    if tmp_bias_data is not None:
        new_conv.bias.data = tmp_bias_data
    #new_conv.weight.to('cuda:0')
    #new_conv.bias.to('cuda:0')
    #print(new_conv.weight.get_device(), new_conv.bias.data, new_conv.bias.get_device())
    return new_conv
