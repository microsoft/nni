# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

import torch
from torch.nn.functional import conv2d

from nni.compression.base.wrapper import ModuleWrapper, register_wrappers

from ..assets.simple_mnist import SimpleTorchModel


def test_wrapper_forward():
    module = torch.nn.Conv2d(3, 3, 3)

    config = {
        'pruning': {
            'target_names': ['_input_', 'weight', 'bias', '_output_'],
            'target_settings': {
                '_input_': {
                    'apply_method': 'mul',
                },
                '_output_': {
                    'apply_method': 'mul',
                },
            },
        },
        'quantization': {
            'target_names': ['_input_', 'weight', '_output_'],
        },
        'distillation': {
            'target_names': ['_output_']
        }
    }
    wrapper = ModuleWrapper(module, '', config)

    # test weight in-channel mask
    mask = torch.ones_like(module.weight).detach()
    mask[:, 1] = 0
    wrapper.update_masks({'weight': mask})
    wrapper.wrap()
    dummy_input = torch.rand(4, 3, 32, 32)
    masked_output = module(dummy_input)
    wrapper.unwrap()
    dummy_input[:, 1] = 0
    simulated_output = module(dummy_input)
    assert torch.equal(masked_output, simulated_output)
    wrapper.pruning_target_spaces['weight'].mask = None

    # test weight out-channel mask
    mask = torch.ones_like(module.weight).detach()
    mask[1, :] = 0
    wrapper.update_masks({'weight': mask})
    wrapper.wrap()
    dummy_input = torch.rand(4, 3, 32, 32)
    masked_output = module(dummy_input)
    wrapper.unwrap()
    simulated_output = module(dummy_input)
    simulated_output[:, 1] = module.bias[1]
    assert torch.equal(masked_output, simulated_output)
    wrapper.pruning_target_spaces['weight'].mask = None

    # test output mask
    mask = torch.ones(3, 30, 30)
    mask[1] = 0
    wrapper.update_masks({'_output_0': mask})
    wrapper.wrap()
    dummy_input = torch.rand(4, 3, 32, 32)
    masked_output = module(dummy_input)
    wrapper.unwrap()
    simulated_output = module(dummy_input)
    simulated_output[:, 1] = 0
    assert torch.equal(masked_output, simulated_output)
    wrapper.pruning_target_spaces['_output_0'].mask = None

    # test quantize weight
    wrapper.quantization_target_spaces['weight'].scale = torch.tensor([0.5])
    wrapper.quantization_target_spaces['weight'].zero_point = torch.tensor([10])
    quant_weight = module.weight.clone().detach()
    quant_weight = (torch.round(torch.clamp(quant_weight / 0.5 + 10, -127, 127)) - 10) * 0.5
    wrapper.wrap()
    dummy_input = torch.rand(4, 3, 32, 32)
    quant_output = module(dummy_input)
    wrapper.unwrap()
    simulated_output = conv2d(dummy_input, quant_weight, module.bias.detach())
    assert torch.equal(quant_output, simulated_output)
    wrapper.quantization_target_spaces['weight'].scale = None
    wrapper.quantization_target_spaces['weight'].zero_point = None

    # test quantize output
    wrapper.quantization_target_spaces['_output_0'].scale = torch.tensor([0.5])
    wrapper.quantization_target_spaces['_output_0'].zero_point = torch.tensor([10])
    wrapper.wrap()
    dummy_input = torch.rand(4, 3, 32, 32)
    quant_output = module(dummy_input)
    wrapper.unwrap()
    simulated_output = module(dummy_input)
    simulated_output = (torch.round(torch.clamp(simulated_output / 0.5 + 10, -127, 127)) - 10) * 0.5
    assert torch.equal(quant_output, simulated_output)
    wrapper.quantization_target_spaces['_output_0'].scale = None
    wrapper.quantization_target_spaces['_output_0'].zero_point = None

    # test distillation observe
    wrapper.distillation_target_spaces['_output_0'].clean()
    wrapper.wrap()
    dummy_input = torch.rand(4, 3, 32, 32)
    output = module(dummy_input)
    wrapper.unwrap()
    hs = wrapper.distillation_target_spaces['_output_0'].hidden_state
    assert torch.equal(output, hs)

    # test fuse compress
    masks = {}
    masks['_input_0'] = torch.ones(3, 32, 32)
    masks['_input_0'][0] = 0
    masks['weight'] = torch.ones_like(module.weight).detach()
    masks['weight'][:, 1] = 0
    masks['weight'][1] = 0
    masks['bias'] = torch.ones_like(module.bias).detach()
    masks['bias'][1] = 0
    masks['_output_0'] = torch.ones(3, 30, 30)
    masks['_output_0'][0] = 0
    wrapper.update_masks(masks)
    wrapper.quantization_target_spaces['_input_0'].scale = torch.tensor([0.4])
    wrapper.quantization_target_spaces['_input_0'].zero_point = torch.tensor([5])
    wrapper.quantization_target_spaces['weight'].scale = torch.tensor([0.5])
    wrapper.quantization_target_spaces['weight'].zero_point = torch.tensor([10])
    wrapper.quantization_target_spaces['_output_0'].scale = torch.tensor([0.3])
    wrapper.quantization_target_spaces['_output_0'].zero_point = torch.tensor([15])
    wrapper.distillation_target_spaces['_output_0'].clean()

    wrapper.wrap()
    dummy_input = torch.rand(4, 3, 32, 32)
    output = module(dummy_input)
    wrapper.unwrap()
    fusion_input = (torch.round(torch.clamp(dummy_input / 0.4 + 5, -127, 127)) - 5) * 0.4
    fusion_input[:, 0] = 0
    fusion_weight = (torch.round(torch.clamp(module.weight.clone().detach() / 0.5 + 10, -127, 127)) - 10) * 0.5
    fusion_weight[1] = 0
    fusion_weight[:, 1] = 0
    fusion_bias = module.bias.clone().detach()
    fusion_bias[1] = 0
    fusion_output = conv2d(fusion_input, fusion_weight, fusion_bias)
    fusion_output = (torch.round(torch.clamp(fusion_output / 0.3 + 15, -127, 127)) - 15) * 0.3
    fusion_output[:, 0] = 0
    assert torch.equal(output, fusion_output)


def test_wrapper_register():
    model = SimpleTorchModel()

    pruning_config_list = [{
        'op_names': ['conv1', 'fc1'],
        'target_names': ['weight', 'bias'],
        'sparse_ratio': 0.4
    }]
    wrappers, _ = register_wrappers(model, pruning_config_list, mode='pruning')
    assert set(wrappers.keys()) == set(['conv1', 'fc1'])

    quantization_config_list = [{
        'op_names': ['conv2', 'fc1'],
        'target_names': ['_input_', 'weight', '_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine'
    }]
    wrappers, _ = register_wrappers(model, quantization_config_list, mode='quantization', existed_wrappers=wrappers)
    assert set(wrappers.keys()) == set(['conv1', 'conv2', 'fc1'])

    distillation_config_list = [{
        'op_names': ['conv3', 'fc1'],
        'target_names': ['_output_'],
        'lambda': 0.1
    }]
    wrappers, _ = register_wrappers(model, distillation_config_list, mode='distillation', existed_wrappers=wrappers)
    assert set(wrappers.keys()) == set(['conv1', 'conv2', 'conv3', 'fc1'])
