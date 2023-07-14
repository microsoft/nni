# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

from collections import defaultdict

import torch

from nni.compression.base.compressor import Pruner, Quantizer

from ..assets.simple_mnist import SimpleTorchModel


def test_compressor():
    # NOTE: not enough, need quantizer-update-calibration-config, distiller
    model = SimpleTorchModel()
    pruning_config_list = [{
        'op_names': ['conv1', 'fc1'],
        'target_names': ['weight', 'bias'],
        'sparse_ratio': 0.4
    }]
    pruner = Pruner(model, pruning_config_list)

    masks = defaultdict(dict)
    masks['conv1']['weight'] = torch.ones_like(model.conv1.weight).detach()
    masks['conv1']['bias'] = torch.ones_like(model.conv1.bias).detach()
    masks['fc1']['weight'] = torch.ones_like(model.fc1.weight).detach()
    masks['fc1']['bias'] = torch.ones_like(model.fc1.bias).detach()
    pruner.update_masks(masks)
    masks = pruner.get_masks()

    quantization_config_list = [{
        'op_names': ['conv2', 'fc1'],
        'target_names': ['_input_', 'weight', '_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine'
    }]
    quantizer = Quantizer.from_compressor(pruner, quantization_config_list)
