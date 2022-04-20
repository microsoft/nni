# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

weighted_modules = [
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'PReLU',
    'Embedding', 'EmbeddingBag',
]


def config_list_canonical(model: Module, config_list: List[Dict]) -> List[Dict]:
    '''
    Split the config by op_names if 'sparsity' or 'sparsity_per_layer' in config,
    and set the sub_config['total_sparsity'] = config['sparsity_per_layer'].
    And every item in 'op_partial_names' will match corresponding 'op_names' in model,
    then convert 'op_partial_names' to 'op_names' in config.

    Example::
        model = models.resnet18()
        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8, 'op_partial_names': ['conv1']}]
        pruner = L1NormPruner(model, config_list)
        pruner.compress()
        pruner.show_pruned_weights()

    In this process, the config_list will implicitly convert to the following:

    [{'op_types': ['Conv2d'], 'sparsity_per_layer': 0.8,
        'op_names': ['conv1', 'layer1.0.conv1', 'layer1.1.conv1',
        'layer2.0.conv1', 'layer2.1.conv1', 'layer3.0.conv1', 'layer3.1.conv1',
        'layer4.0.conv1', 'layer4.1.conv1']}]
    '''
    config_list = deepcopy(config_list)

    for config in config_list:
        if 'sparsity' in config:
            if 'sparsity_per_layer' in config:
                raise ValueError("'sparsity' and 'sparsity_per_layer' have the same semantics, can not set both in one config.")
            else:
                config['sparsity_per_layer'] = config.pop('sparsity')

    for config in config_list:
        if 'op_types' in config:
            if 'default' in config['op_types']:
                config['op_types'].remove('default')
                config['op_types'].extend(weighted_modules)

    for config in config_list:
        if 'op_partial_names' in config:
            op_names = []
            for partial_name in config['op_partial_names']:
                for name, _ in model.named_modules():
                    if partial_name in name:
                        op_names.append(name)
            if 'op_names' in config:
                config['op_names'].extend(op_names)
                config['op_names'] = list(set(config['op_names']))
            else:
                config['op_names'] = op_names
            config.pop('op_partial_names')

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
    """
    Compare origin model and compact model, return the sparsity of each group mentioned in config list.
    A group means all layer mentioned in one config.
    e.g., a linear named 'linear1' and its weight size is [100, 100] in origin model, but in compact model,
    the layer weight size with same layer name is [100, 50],
    then this function will return [{'op_names': 'linear1', 'total_sparsity': 0.5}].
    """
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


def compute_sparsity_mask2compact(compact_model: Module, compact_model_masks: Dict[str, Dict[str, Tensor]], config_list: List[Dict]):
    """
    Apply masks on compact model, return the sparsity of each group mentioned in config list.
    A group means all layer mentioned in one config.
    This function count all zero elements of the masks in one group,
    then divide by the elements number of the weights in this group to compute sparsity.
    """
    mask2compact_sparsity = []
    for config in config_list:
        left_weight_num = 0
        total_weight_num = 0
        for module_name, module in compact_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            module_weight_num = module.weight.data.numel()
            total_weight_num += module_weight_num
            if module_name in compact_model_masks:
                weight_mask = compact_model_masks[module_name]['weight']
                left_weight_num += len(torch.nonzero(weight_mask, as_tuple=False))
            else:
                left_weight_num += module_weight_num
        mask2compact_sparsity.append(deepcopy(config))
        mask2compact_sparsity[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return mask2compact_sparsity


def compute_sparsity(origin_model: Module, compact_model: Module, compact_model_masks: Dict[str, Dict[str, Tensor]],
                     config_list: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    This function computes how much the origin model has been compressed in the current state.
    The current state means `compact_model` + `compact_model_masks`
    (i.e., `compact_model_masks` applied on `compact_model`).
    The compact model is the origin model after pruning,
    and it may have different structure with origin_model cause of speed up.

    Returns
    -------
    Tuple[List[Dict], List[Dict], List[Dict]]
        (current2origin_sparsity, compact2origin_sparsity, mask2compact_sparsity).
        current2origin_sparsity is how much the origin model has been compressed in the current state.
        compact2origin_sparsity is the sparsity obtained by comparing the structure of origin model and compact model.
        mask2compact_sparsity is the sparsity computed by count the zero value in the mask.
    """
    compact2origin_sparsity = compute_sparsity_compact2origin(origin_model, compact_model, config_list)
    mask2compact_sparsity = compute_sparsity_mask2compact(compact_model, compact_model_masks, config_list)
    assert len(compact2origin_sparsity) == len(mask2compact_sparsity), 'Length mismatch.'
    current2origin_sparsity = []
    for c2o_sparsity, m2c_sparsity, config in zip(compact2origin_sparsity, mask2compact_sparsity, config_list):
        current2origin_sparsity.append(deepcopy(config))
        current2origin_sparsity[-1]['total_sparsity'] = 1 - (1 - c2o_sparsity['total_sparsity']) * (1 - m2c_sparsity['total_sparsity'])
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
            if module_name in masks and isinstance(masks[module_name]['weight'], Tensor):
                weight_mask = masks[module_name]['weight']
                masked_rate[module_name] = 1 - (weight_mask.sum().item() / weight_mask.numel())
                model_weights_numel[module_name] = round(weight_mask.sum().item())
            else:
                model_weights_numel[module_name] = module.weight.data.numel()
    return model_weights_numel, masked_rate


# FIXME: to avoid circular import, copy this function in this place
def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name
    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module
    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None
