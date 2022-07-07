# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from operator import is_

import pytest
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

import nni
from nni.compression.pytorch.pruning import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    SlimPruner,
    FPGMPruner,
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
    TaylorFOWeightPruner,
    ADMMPruner,
    MovementPruner
)

from ..assets.device import device
from ..assets.simple_mnist import (
    SimpleLightningModel,
    SimpleTorchModel,
    create_lighting_evaluator,
    create_pytorch_evaluator,
    evaluating_model,
    training_model
)
from ..assets.utils import unfold_config_list


def create_model(model_type: str):
    torch_config_list = [{'op_types': ['Linear'], 'sparsity': 0.5},
                         {'op_names': ['conv1', 'conv2', 'conv3'], 'sparsity': 0.5},
                         {'op_names': ['fc2'], 'exclude': True}]

    lightning_config_list = [{'op_types': ['Linear'], 'sparsity': 0.5},
                             {'op_names': ['model.conv1', 'model.conv2', 'model.conv3'], 'sparsity': 0.5},
                             {'op_names': ['model.fc2'], 'exclude': True}]

    if model_type == 'lightning':
        model = SimpleLightningModel()
        config_list = lightning_config_list
        dummy_input = torch.rand(8, 1, 28, 28)
    elif model_type == 'pytorch':
        model = SimpleTorchModel().to(device)
        config_list = torch_config_list
        dummy_input = torch.rand(8, 1, 28, 28, device=device)
    else:
        raise ValueError(f'wrong model_type: {model_type}')
    return model, config_list, dummy_input


def validate_masks(masks: Dict[str, Dict[str, torch.Tensor]], model: torch.nn.Module, config_list: List[Dict[str, Any]],
                   is_global: bool = False):
    config_dict = unfold_config_list(model, config_list)
    # validate if all configured layers have generated mask.
    mismatched_op_names = set(config_dict.keys()).symmetric_difference(masks.keys())
    assert f'mismatched op_names: {mismatched_op_names}'

    target_name = 'weight'
    total_masked_numel = 0
    total_target_numel = 0
    for module_name, target_masks in masks.items():
        mask = target_masks[target_name]
        assert mask.numel() == (mask == 0).sum().item() + (mask == 1).sum().item(), f'{module_name} {target_name} mask has values other than 0 and 1.'
        if not is_global:
            excepted_sparsity = config_dict[module_name].get('sparsity', config_dict[module_name].get('total_sparsity'))
            real_sparsity = (mask == 0).sum().item() / mask.numel()
            err_msg = f'{module_name} {target_name} excepted sparsity: {excepted_sparsity}, but real sparsity: {real_sparsity}'
            assert excepted_sparsity * 0.99 < real_sparsity < excepted_sparsity * 1.01, err_msg
        else:
            total_masked_numel += (mask == 0).sum().item()
            total_target_numel += mask.numel()
    if is_global:
        excepted_sparsity = next(iter(config_dict.values())).get('sparsity', config_dict[module_name].get('total_sparsity'))
        real_sparsity = total_masked_numel / total_target_numel
        err_msg = f'excepted global sparsity: {excepted_sparsity}, but real global sparsity: {real_sparsity}.'
        assert excepted_sparsity * 0.9 < real_sparsity < excepted_sparsity * 1.1, err_msg


def validate_dependency_aware(model_type: str, masks: Dict[str, Dict[str, torch.Tensor]]):
    # only for simple_mnist model
    if model_type == 'lightning':
        assert torch.equal(masks['model.conv2']['weight'].mean([1, 2, 3]), masks['model.conv3']['weight'].mean([1, 2, 3]))
    if model_type == 'pytorch':
        assert torch.equal(masks['conv2']['weight'].mean([1, 2, 3]), masks['conv3']['weight'].mean([1, 2, 3]))


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
def test_level_pruner(model_type: str):
    model, config_list, dummy_input = create_model(model_type)

    pruner = LevelPruner(model=model, config_list=config_list)

    _, masks = pruner.compress()
    model(dummy_input)
    pruner._unwrap_model()
    validate_masks(masks, model, config_list)


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('pruning_type', ['l1', 'l2', 'fpgm'])
@pytest.mark.parametrize('mode', ['normal', 'dependency_aware'])
def test_norm_pruner(model_type: str, pruning_type: str, mode: str):
    model, config_list, dummy_input = create_model(model_type)

    if pruning_type == 'l1':
        pruner = L1NormPruner(model=model, config_list=config_list, mode=mode, dummy_input=dummy_input)
    elif pruning_type == 'l2':
        pruner = L2NormPruner(model=model, config_list=config_list, mode=mode, dummy_input=dummy_input)
    elif pruning_type == 'fpgm':
        pruner = FPGMPruner(model=model, config_list=config_list, mode=mode, dummy_input=dummy_input)
    else:
        raise ValueError(f'wrong norm: {pruning_type}')

    _, masks = pruner.compress()
    model(dummy_input)
    pruner._unwrap_model()
    validate_masks(masks, model, config_list)
    if mode == 'dependency_aware':
        validate_dependency_aware(model_type, masks)


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('using_evaluator', [True, False])
@pytest.mark.parametrize('mode', ['global', 'normal'])
def test_slim_pruner(model_type: str, using_evaluator: bool, mode: str):
    model, _, dummy_input = create_model(model_type)
    config_list = [{'op_types': ['BatchNorm2d'], 'total_sparsity': 0.5}]

    if using_evaluator:
        evaluator = create_lighting_evaluator() if model_type == 'lightning' else create_pytorch_evaluator(model)
        pruner = SlimPruner(model=model, config_list=config_list, evaluator=evaluator, training_epochs=1, scale=0.0001, mode=mode)
    else:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        pruner = SlimPruner(model=model, config_list=config_list, trainer=training_model, traced_optimizer=optimizer,
                            criterion=F.nll_loss, training_epochs=1, scale=0.0001, mode=mode)

    _, masks = pruner.compress()
    model(dummy_input)
    pruner._unwrap_model()
    validate_masks(masks, model, config_list, is_global=(mode == 'global'))


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('pruning_type', ['apoz', 'mean', 'taylor'])
@pytest.mark.parametrize('using_evaluator', [True, False])
@pytest.mark.parametrize('mode', ['normal', 'dependency_aware'])
def test_hook_based_pruner(model_type: str, pruning_type: str, using_evaluator: bool, mode: str):
    model, config_list, dummy_input = create_model(model_type)

    if using_evaluator:
        evaluator = create_lighting_evaluator() if model_type == 'lightning' else create_pytorch_evaluator(model)
        if pruning_type == 'apoz':
            pruner = ActivationAPoZRankPruner(model=model, config_list=config_list, evaluator=evaluator, training_steps=20,
                                              activation='relu', mode=mode, dummy_input=dummy_input)
        elif pruning_type == 'mean':
            pruner = ActivationMeanRankPruner(model=model, config_list=config_list, evaluator=evaluator, training_steps=20,
                                              activation='relu', mode=mode, dummy_input=dummy_input)
        elif pruning_type == 'taylor':
            pruner = TaylorFOWeightPruner(model=model, config_list=config_list, evaluator=evaluator, training_steps=20,
                                          mode=mode, dummy_input=dummy_input)
    else:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        if pruning_type == 'apoz':
            pruner = ActivationAPoZRankPruner(model=model, config_list=config_list, trainer=training_model, traced_optimizer=optimizer,
                                              criterion=F.nll_loss, training_batches=20, activation='relu', mode=mode, dummy_input=dummy_input)
        elif pruning_type == 'mean':
            pruner = ActivationMeanRankPruner(model=model, config_list=config_list, trainer=training_model, traced_optimizer=optimizer,
                                              criterion=F.nll_loss, training_batches=20, activation='relu', mode=mode, dummy_input=dummy_input)
        elif pruning_type == 'taylor':
            pruner = TaylorFOWeightPruner(model=model, config_list=config_list, trainer=training_model, traced_optimizer=optimizer,
                                          criterion=F.nll_loss, training_batches=20, mode=mode, dummy_input=dummy_input)

    _, masks = pruner.compress()
    model(dummy_input)
    pruner._unwrap_model()
    validate_masks(masks, model, config_list)
    if mode == 'dependency_aware':
        validate_dependency_aware(model_type, masks)
