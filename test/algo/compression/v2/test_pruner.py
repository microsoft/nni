# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

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
    create_lighting_evaluator,
    create_pytorch_evaluator,
    training_model
)
from ..assets.common import create_model, validate_masks, validate_dependency_aware


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


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('using_evaluator', [True, False])
@pytest.mark.parametrize('granularity', ['fine-grained', 'coarse-grained'])
def test_admm_pruner(model_type: str, using_evaluator: bool, granularity: str):
    model, config_list, dummy_input = create_model(model_type)

    if using_evaluator:
        evaluator = create_lighting_evaluator() if model_type == 'lightning' else create_pytorch_evaluator(model)
        pruner = ADMMPruner(model=model, config_list=config_list, evaluator=evaluator, iterations=2, training_epochs=1, granularity=granularity)
    else:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        pruner = ADMMPruner(model=model, config_list=config_list, trainer=training_model, traced_optimizer=optimizer, criterion=F.nll_loss,
                            iterations=2, training_epochs=1, granularity=granularity)

    _, masks = pruner.compress()
    model(dummy_input)
    pruner._unwrap_model()
    validate_masks(masks, model, config_list)


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('using_evaluator', [True, False])
def test_movement_pruner(model_type: str, using_evaluator: bool):
    model, config_list, dummy_input = create_model(model_type)

    if using_evaluator:
        evaluator = create_lighting_evaluator() if model_type == 'lightning' else create_pytorch_evaluator(model)
        pruner = MovementPruner(model=model, config_list=config_list, evaluator=evaluator, training_epochs=1, warm_up_step=10, cool_down_beginning_step=40)
    else:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        pruner = MovementPruner(model=model, config_list=config_list, trainer=training_model, traced_optimizer=optimizer, criterion=F.nll_loss,
                                training_epochs=1, warm_up_step=10, cool_down_beginning_step=40)

    _, masks = pruner.compress()
    model(dummy_input)
    pruner._unwrap_model()
    validate_masks(masks, model, config_list)
