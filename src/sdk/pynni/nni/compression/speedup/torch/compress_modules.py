# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .infer_shape import CoarseMask

compress_modules = {
    'BatchNorm2d': lambda module, mask: compress_batchnorm2d(module, mask),
    'Conv2d': lambda module, mask: compress_conv2d(module, mask)
}

cms_output = {
    'BatchNorm2d': lambda module, output_cmask: compress_batchnorm2d_output(module, output_cmask),
    'Conv2d': lambda module, output_cmask: compress_conv2d_output(module, output_cmask)
}


def compress_batchnorm2d_output(module, output_cmask):
    """
    """

def compress_batchnorm2d(norm, mask):
    """
    """
    assert 'weight' in mask and 'bias' in mask
    sum_mask = mask['weight'] + mask['bias']
    nonzero_index = torch.nonzero(sum_mask, as_tuple=True)[0]
    new_norm = torch.nn.BatchNorm2d(num_features=nonzero_index.size()[0],
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, nonzero_index)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, nonzero_index)
    if norm.track_running_stats:
        new_norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, nonzero_index)
        new_norm.running_var.data = torch.index_select(norm.running_var.data, 0, nonzero_index)
    # infer shape of input tensor
    input_cmask = CoarseMask(num_dim=4)
    input_cmask.add_index_mask(dim=1,
                               index=torch.nonzero(mask['weight'], as_tuple=True)[0])
    # infer shape of output tensor
    output_cmask = CoarseMask(num_dim=4)
    output_cmask.add_index_mask(dim=1, index=nonzero_index)
    return new_norm, input_cmask, output_cmask

def compress_conv2d_output(module, output_cmask):
    """
    """

def compress_conv2d(conv, mask):
    """
    """
    # fine-grained tensor sparse
    #...
    # coarse-grained shape sparse
    #...
