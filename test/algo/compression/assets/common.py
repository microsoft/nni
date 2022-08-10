# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Any, Dict, List

import torch

from .device import device
from .simple_mnist import SimpleLightningModel, SimpleTorchModel
from .utils import unfold_config_list


log_dir = Path(__file__).parent.parent / 'logs'


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
            assert excepted_sparsity * 0.9 < real_sparsity < excepted_sparsity * 1.1, err_msg
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
