# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module


def config_list_canonical(model: Module, config_list: List[Dict]) -> List[Dict]:
    '''
    Split the config by op_names if 'sparsity' or 'sparsity_per_layer' in config,
    and set the sub_config['total_sparsity'] = config['sparsity_per_layer'].
    '''
    for config in config_list:
        if 'sparsity' in config:
            if 'sparsity_per_layer' in config:
                raise ValueError("'sparsity' and 'sparsity_per_layer' have the same semantics, can not set both in one config.")
            else:
                config['sparsity_per_layer'] = config.pop('sparsity')

    config_list = dedupe_config_list(unfold_config_list(model, config_list))
    new_config_list = []

    for config in config_list:
        if 'sparsity_per_layer' in config:
            sparsity_per_layer = config.pop('sparsity_per_layer')
            op_names = config.pop('op_names')
            for op_name in op_names:
                sub_config = deepcopy(config)
                sub_config['op_names'] = [op_name]
                sub_config['total_sparsity'] = sparsity_per_layer
                new_config_list.append(sub_config)
        elif 'max_sparsity_per_layer' in config and isinstance(config['max_sparsity_per_layer'], float):
            op_names = config.get('op_names', [])
            max_sparsity_per_layer = {}
            max_sparsity = config['max_sparsity_per_layer']
            for op_name in op_names:
                max_sparsity_per_layer[op_name] = max_sparsity
            config['max_sparsity_per_layer'] = max_sparsity_per_layer
            new_config_list.append(config)
        else:
            new_config_list.append(config)

    return new_config_list


def unfold_config_list(model: Module, config_list: List[Dict]) -> List[Dict]:
    '''
    Unfold config_list to op_names level.
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
    Dedupe the op_names in unfolded config_list.
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


def compute_sparsity_compact2origin(origin_model: Module, compact_model: Module, config_list: List[Dict]) -> List[Dict]:
    compact2origin_sparsity = []
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
        compact2origin_sparsity.append(deepcopy(config))
        compact2origin_sparsity[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return compact2origin_sparsity


def compute_sparsity_mask2compact(masked_model: Module, masks: Dict[str, Dict[str, Tensor]], config_list: List[Dict]):
    mask2compact_sparsity = []
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
        mask2compact_sparsity.append(deepcopy(config))
        mask2compact_sparsity[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return mask2compact_sparsity


def compute_sparsity(origin_model: Module, compact_model: Module, masks: Dict[str, Dict[str, Tensor]],
                     config_list: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    compact2origin_sparsity = compute_sparsity_compact2origin(origin_model, compact_model, config_list)
    mask2compact_sparsity = compute_sparsity_mask2compact(compact_model, masks, config_list)
    assert len(compact2origin_sparsity) == len(mask2compact_sparsity), 'Length mismatch.'
    current2origin_sparsity = []
    for mo_sparsity, ms_sparsity, config in zip(compact2origin_sparsity, mask2compact_sparsity, config_list):
        current2origin_sparsity.append(deepcopy(config))
        current2origin_sparsity[-1]['total_sparsity'] = 1 - (1 - mo_sparsity['total_sparsity']) * (1 - ms_sparsity['total_sparsity'])
    return current2origin_sparsity, compact2origin_sparsity, mask2compact_sparsity


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
