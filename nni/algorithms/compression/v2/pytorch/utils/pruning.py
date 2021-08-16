# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Dict, List

from torch.nn import Module


def config_list_canonical(model: Module, config_list: List[Dict]) -> List[Dict]:
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
                new_config = deepcopy(config)
                new_config['op_names'] = [op_name]
                new_config['total_sparsity'] = sparsity_per_layer
                new_config_list.append(new_config)
        else:
            new_config_list.append(config)

    return new_config_list


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
