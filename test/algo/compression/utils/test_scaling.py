# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

import torch

from nni.compression.utils.scaling import Scaling


@pytest.mark.parametrize("kernel_padding_mode", ['front', 'back'])
@pytest.mark.parametrize("kernel_padding_val", [None, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_scaling(kernel_padding_mode, kernel_padding_val, keepdim):
    data = torch.arange(0, 24, requires_grad=True, dtype=torch.float32).reshape([2, 3, 4])

    if keepdim:
        if kernel_padding_mode == 'back' and kernel_padding_val in [None, -1]:
            data_mean = data.detach().mean(-1, keepdim=True).mean(-2, keepdim=True)
        elif kernel_padding_mode == 'front' and kernel_padding_val in [-1]:
            data_mean = data.detach().mean(0, keepdim=True).mean(1, keepdim=True)
        else:
            data_mean = data.detach().clone()
    else:
        if kernel_padding_mode == 'back' and kernel_padding_val in [None, -1]:
            data_mean = data.detach().mean(-1).mean(-1)
        elif kernel_padding_mode == 'front' and kernel_padding_val in [-1]:
            data_mean = data.detach().mean(0).mean(0)
        else:
            data_mean = data.detach().clone()

    def reduce_func(tensor: torch.Tensor):
        return tensor.mean(-1)
    
    def reduce_func_detach(tensor: torch.Tensor):
        return tensor.detach().mean(-1)

    scaler = Scaling(kernel_size=[1], kernel_padding_mode=kernel_padding_mode, kernel_padding_val=kernel_padding_val)

    shinked_data = scaler.shrink(data, reduce_func, keepdim=keepdim)
    assert shinked_data.requires_grad and torch.equal(data_mean, shinked_data)
    expand_data = scaler.expand(shinked_data, expand_size=[2, 3, 4], keepdim=keepdim)
    assert expand_data.requires_grad and list(expand_data.shape) == [2, 3, 4]

    shinked_data = scaler.shrink(data, reduce_func_detach, keepdim=keepdim)
    assert not shinked_data.requires_grad and torch.equal(data_mean, shinked_data)
    expand_data = scaler.expand(shinked_data, expand_size=[2, 3, 4], keepdim=keepdim)
    assert not expand_data.requires_grad and list(expand_data.shape) == [2, 3, 4]
