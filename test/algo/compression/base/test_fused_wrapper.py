# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest
import sys
from copy import deepcopy

import torch
import torch.nn.functional as F

from nni.compression.base.wrapper import IdentityModuleWrapper, register_wrappers
from ..assets.simple_mnist import SimpleTorchModel

def test_fused_wrapper_register():
    model = SimpleTorchModel()
    config_list_1 = [
    {
        'op_names': ['fc1', "conv1"],
        'target_names': ['_input_', '_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine',
        'granularity': 'default'
    }]

    config_list_2 = [{
        'op_names': ['conv1', 'fc2', 'conv2'],
        'target_names': ['weight', '_input_', "_output_"],
        'quant_dtype': 'int2',
        'quant_scheme': 'affine',
        'granularity': 'default',
        'fuse_names': [("conv1", "bn1"), ("conv2", "bn2")]
    }]

    wrappers_1, _ = register_wrappers(model, config_list_1, mode='quantization')
    assert set(wrappers_1.keys()) == set(['fc1', 'conv1'])

    wrappers_2, _ =register_wrappers(model, config_list_2, 'quantization', wrappers_1)
    assert set(wrappers_2.keys()) == set(['fc1', 'conv1', 'fc2', 'conv2', 'bn1', 'bn2'])
    for module_name, wrapper in wrappers_2.items():
        wrapper.wrap()
        if module_name in ['conv1', 'conv2']:
            assert hasattr(wrapper, 'is_bias') and hasattr(wrapper, 'bias') \
                and getattr(wrapper, 'is_bias') == 'Tensor' and isinstance(wrapper.bias, torch.Tensor)
        elif module_name in ['bn1', 'bn2']:
            assert isinstance(wrapper, IdentityModuleWrapper)




