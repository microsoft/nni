# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from typing import Dict, Type

import torch

from nni.contrib.compression.pruning import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    SlimPruner,
    TaylorPruner,
    MovementPruner,
    LinearPruner,
    AGPPruner
)
from nni.contrib.compression.base.compressor import Pruner
from nni.contrib.compression.utils import auto_set_denpendency_group_ids

from nni.compression.pytorch.speedup.v2 import ModelSpeedup

from ..assets.common import create_model
from ..assets.simple_mnist import SimpleLightningModel, SimpleTorchModel, create_lighting_evaluator, create_pytorch_evaluator


pruner_dict: Dict[str, Type[Pruner]] = {
    'level': LevelPruner,
    'l1': L1NormPruner,
    'l2': L2NormPruner,
    'fpgm': FPGMPruner,
    'slim': SlimPruner,
    'taylor': TaylorPruner,
    'mvp': MovementPruner,
    'linear': LinearPruner,
    'agp': AGPPruner
}


def check_masks(masks: Dict[str, Dict[str, torch.Tensor]], model_type, prune_type):
    prefix = 'model.' if model_type == 'lightning' else ''
    assert 1 - masks[f'{prefix}conv1']['weight'].sum() / masks[f'{prefix}conv1']['weight'].numel() == 0.75
    assert 1 - masks[f'{prefix}conv2']['weight'].sum() / masks[f'{prefix}conv2']['weight'].numel() == 0.75
    assert 1 - masks[f'{prefix}conv3']['weight'].sum() / masks[f'{prefix}conv3']['weight'].numel() == 0.75
    assert 1 - masks[f'{prefix}fc1']['weight'].sum() / masks[f'{prefix}fc1']['weight'].numel() == 0.75

    assert torch.equal(masks[f'{prefix}conv1']['weight'].sum([1, 2, 3]).bool().float(), masks[f'{prefix}conv1']['bias'])
    assert torch.equal(masks[f'{prefix}conv2']['weight'].sum([1, 2, 3]).bool().float(), masks[f'{prefix}conv2']['bias'])
    assert torch.equal(masks[f'{prefix}conv3']['weight'].sum([1, 2, 3]).bool().float(), masks[f'{prefix}conv3']['bias'])
    assert torch.equal(masks[f'{prefix}fc1']['weight'].sum([1]).bool().float(), masks[f'{prefix}fc1']['bias'])
    assert torch.equal(masks[f'{prefix}conv2']['weight'].sum([1, 2, 3]).bool().float(), masks[f'{prefix}conv3']['weight'].sum([1, 2, 3]).bool().float())


def check_pruned_simple_model(model: torch.nn.Module, ori_model: torch.nn.Module, dummy_input, model_type):
    model(dummy_input)
    if model_type == 'lightning':
        model = model.model
        ori_model = ori_model.model
    assert 1 - model.conv1.weight.shape[0] / ori_model.conv1.weight.shape[0] == 0.75
    assert 1 - model.conv2.weight.shape[0] / ori_model.conv2.weight.shape[0] == 0.75
    assert 1 - model.conv3.weight.shape[0] / ori_model.conv3.weight.shape[0] == 0.75
    assert 1 - model.fc1.weight.shape[0] / ori_model.fc1.weight.shape[0] == 0.75
    

@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('prune_type', ['l1', 'l2', 'fpgm'])
def test_norm_pruner(model_type: str, prune_type: str):
    model, config_list_dict, dummy_input = create_model(model_type)

    config_list = config_list_dict['pruning']
    # set dependency group ids if not level
    config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
    pruner = pruner_dict[prune_type](model, config_list)
    _, masks = pruner.compress()
    pruner.unwrap_model()

    check_masks(masks, model_type, prune_type)

    pruned_model = ModelSpeedup(model, dummy_input, masks).speedup_model()
    ori_model, _, _  = create_model(model_type)
    check_pruned_simple_model(pruned_model, ori_model, dummy_input, model_type)

