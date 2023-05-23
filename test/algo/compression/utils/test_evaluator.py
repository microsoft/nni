# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

import torch

from nni.compression.utils.evaluator import (
    TensorHook,
    ForwardHook,
    BackwardHook,
)

from ..assets.device import device
from ..assets.simple_mnist import (
    SimpleLightningModel,
    SimpleTorchModel,
    create_lighting_evaluator,
    create_pytorch_evaluator
)


optimizer_before_step_flag = False
optimizer_after_step_flag = False
loss_flag = False

def optimizer_before_step_patch():
    global optimizer_before_step_flag
    optimizer_before_step_flag = True

def optimizer_after_step_patch():
    global optimizer_after_step_flag
    optimizer_after_step_flag = True

def loss_patch(t: torch.Tensor, batch):
    global loss_flag
    loss_flag = True
    return t

def tensor_hook_factory(buffer: list):
    def hook_func(t: torch.Tensor):
        buffer.append(True)
    return hook_func

def forward_hook_factory(buffer: list):
    def hook_func(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        buffer.append(True)
    return hook_func

def backward_hook_factory(buffer: list):
    def hook_func(module: torch.nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
        buffer.append(True)
    return hook_func

def reset_flags():
    global optimizer_before_step_flag, optimizer_after_step_flag, loss_flag
    optimizer_before_step_flag = False
    optimizer_after_step_flag = False
    loss_flag = False

def assert_flags():
    global optimizer_before_step_flag, optimizer_after_step_flag, loss_flag
    assert optimizer_before_step_flag, 'Evaluator patch optimizer before step failed.'
    assert optimizer_after_step_flag, 'Evaluator patch optimizer after step failed.'
    assert loss_flag, 'Evaluator patch loss failed.'


@pytest.mark.parametrize("evaluator_type", ['lightning', 'pytorch'])
def test_evaluator(evaluator_type: str):
    if evaluator_type == 'lightning':
        model = SimpleLightningModel()
        evaluator = create_lighting_evaluator()
        evaluator._init_optimizer_helpers(model)
        evaluator.bind_model(model)
        tensor_hook = TensorHook(model.model.conv1.weight, 'model.conv1.weight', tensor_hook_factory)
        forward_hook = ForwardHook(model.model.conv1, 'model.conv1', forward_hook_factory)
        backward_hook = BackwardHook(model.model.conv1, 'model.conv1', backward_hook_factory)
    elif evaluator_type == 'pytorch':
        model = SimpleTorchModel().to(device)
        evaluator = create_pytorch_evaluator(model)
        evaluator._init_optimizer_helpers(model)
        evaluator.bind_model(model)
        tensor_hook = TensorHook(model.conv1.weight, 'conv1.weight', tensor_hook_factory)
        forward_hook = ForwardHook(model.conv1, 'conv1', forward_hook_factory)
        backward_hook = BackwardHook(model.conv1, 'conv1', backward_hook_factory)
    else:
        raise ValueError(f'wrong evaluator_type: {evaluator_type}')

    # test train with patch & hook
    reset_flags()
    evaluator.patch_loss(loss_patch)
    evaluator.patch_optimizer_step([optimizer_before_step_patch], [optimizer_after_step_patch])
    evaluator.register_hooks([tensor_hook, forward_hook, backward_hook])

    evaluator.train(max_steps=1)
    assert_flags()
    assert all([len(hook.buffer) == 1 for hook in [tensor_hook, forward_hook, backward_hook]])

    # test finetune with patch & hook
    reset_flags()
    evaluator.remove_all_hooks()
    evaluator.register_hooks([tensor_hook, forward_hook, backward_hook])

    evaluator.finetune()
    assert_flags()
    assert all([len(hook.buffer) == 50 for hook in [tensor_hook, forward_hook, backward_hook]])
