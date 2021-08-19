# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module


def apply_compression_results(model: Module, masks: Dict[str, Dict[str, Tensor]]):
    """
    Note: this function is for inference, because it simply multiplies weights with
    corresponding masks when this function is called.
    Parameters
    ----------
    model
        The model to be sparsified.
    masks
        The masks of the model.
    """
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data = module.weight.data.mul_(masks[name]['weight_mask'])
            if hasattr(module, 'bias') and module.bias is not None and 'bias_mask' in masks[name]:
                module.bias.data = module.bias.data.mul_(masks[name]['bias_mask'])


def unfold_config_list(model: Module, config_list: List[Dict]) -> List[Dict]:
    '''
    unfold config_list to op_names level
    '''
    unfolded_config_list = []
    for config in config_list:
        op_names = []
        for module_name, module in model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            op_names.append(module_name)
        unfolded_config = deepcopy(config)
        unfolded_config['op_names'] = op_names
        unfolded_config_list.append(unfolded_config)
    return unfolded_config_list


def dedupe_config_list(config_list: List[Dict]) -> List[Dict]:
    '''
    dedupe the op_names in unfolded config_list
    '''
    exclude = set()
    exclude_idxes = []
    config_list = deepcopy(config_list)
    for idx, config in reversed(list(enumerate(config_list))):
        if 'exclude' in config:
            exclude.update(config['op_names'])
            exclude_idxes.append(idx)
            continue
        config['op_names'] = sorted(list(set(config['op_names']).difference(exclude)))
        exclude.update(config['op_names'])
    for idx in sorted(exclude_idxes, reverse=True):
        config_list.pop(idx)
    return config_list


def compute_sparsity_with_compact_model(origin_model: Module, compact_model: Module, config_list: List[Dict]) -> List[Dict]:
    real_config_list = []
    for config in config_list:
        left_weight_num = 0
        total_weight_num = 0
        for module_name, module in origin_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            total_weight_num += module.weight.data.numel()
        for module_name, module in compact_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            left_weight_num += module.weight.data.numel()
        real_config_list.append(deepcopy(config))
        real_config_list[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return real_config_list


def compute_sparsity_with_masks(masked_model: Module, masks: Dict[str, Dict[str, Tensor]], config_list: List[Dict]):
    real_config_list = []
    for config in config_list:
        left_weight_num = 0
        total_weight_num = 0
        for module_name, module in masked_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            module_weight_num = module.weight.data.numel()
            total_weight_num += module_weight_num
            if module_name in masks:
                weight_mask = masks[module_name]['weight_mask']
                left_weight_num += len(torch.nonzero(weight_mask, as_tuple=False))
            else:
                left_weight_num += module_weight_num
        real_config_list.append(deepcopy(config))
        real_config_list[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return real_config_list


def compute_sparsity(origin_model: Module, compact_model: Module, masks: Dict[str, Dict[str, Tensor]],
                     config_list: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    model_based_sparsity = compute_sparsity_with_compact_model(origin_model, compact_model, config_list)
    masks_based_sparsity = compute_sparsity_with_masks(compact_model, masks, config_list)
    assert len(model_based_sparsity) == len(masks_based_sparsity), 'Length mismatch.'
    real_config_list = []
    for mo_sparsity, ms_sparsity, config in zip(model_based_sparsity, masks_based_sparsity, config_list):
        real_config_list.append(deepcopy(config))
        real_config_list[-1]['total_sparsity'] = 1 - (1 - mo_sparsity['total_sparsity']) * (1 - ms_sparsity['total_sparsity'])
    return real_config_list, model_based_sparsity, masks_based_sparsity


def get_model_weights_numel(model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]] = {}) -> Dict:
    """
    Count the layer weight elements number in config_list.
    If masks is not empty, the masked weight will not be counted.
    """
    model_weights_numel = {}
    masked_rate = {}
    for config in config_list:
        for module_name, module in model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            if module_name in masks and isinstance(masks[module_name]['weight_mask'], Tensor):
                weight_mask = masks[module_name]['weight_mask']
                masked_rate[module_name] = 1 - (weight_mask.sum().item() / weight_mask.numel())
                model_weights_numel[module_name] = round(weight_mask.sum().item())
            else:
                model_weights_numel[module_name] = module.weight.data.numel()
    return model_weights_numel, masked_rate
