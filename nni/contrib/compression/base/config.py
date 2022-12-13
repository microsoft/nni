# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Dict, List, Tuple

import torch

from .setting import INPUT_PREFIX, OUTPUT_PREFIX


def trans_legacy_config_list(config_list: List[Dict[str, Any]],
                             default_sparse_granularity: str | List[int] | None = None) -> List[Dict[str, Any]]:
    # transfer the old config keys to new config keys
    if not config_list:
        return config_list

    if 'exclude' not in config_list[0] and 'sparsity' not in config_list[0] and 'sparsity_per_layer' \
        not in config_list[0] and 'total_sparsity' not in config_list[0] and 'quant_types' not in config_list[0]:
        # all the old keys not in the first config, then we assume this config list is not a legacy config list.
        return config_list

    config_list = deepcopy(config_list)

    # trans common part
    ex_op_types = set()
    ex_op_names = set()
    ex_op_names_re = set()
    for config in config_list:
        if 'op_partial_names' in config:
            config['op_names_re'] = [f'.*{p_name}.*' for p_name in config.pop('op_partial_names')]
        if 'exclude' in config:
            config.pop('exclude')
            ex_op_types.update(config.get('op_types'))
            ex_op_names.update(config.get('op_names'))
            ex_op_names_re.update(config.get('op_names_re'))
    if ex_op_types or ex_op_names or ex_op_names_re:
        for config in config_list:
            config['exclude_op_types'] = list(ex_op_types)
            config['exclude_op_names'] = list(ex_op_names)
            config['exclude_op_names_re'] = list(ex_op_names_re)

    # trans pruning config
    group_id_candidate = 0
    for config in config_list:
        sparse_ratio = None
        group_id = None
        max_sparse_ratio = config.pop('max_sparsity_per_layer', None)
        if 'sparsity_per_layer' in config or 'sparsity' in config:
            sparse_ratio = config.pop('sparsity_per_layer', config.pop('sparsity'))
        if 'total_sparsity' in config:
            sparse_ratio = config.pop('total_sparsity')
            group_id = group_id_candidate
            group_id_candidate += 1
        if sparse_ratio is not None:
            config['target_names'] = ['weight', 'bias']
            config['target_settings'] = {
                'weight': {
                    'sparse_ratio': sparse_ratio,
                    'max_sparse_ratio': max_sparse_ratio,
                    'global_group_id': group_id,
                    'sparse_granularity': default_sparse_granularity,
                }
            }

    # trans quantization part
    for config in config_list:
        target_names = []
        target_settings = {}
        if 'quant_types' in config:
            quant_types, quant_bits = config.pop('quant_types'), config.pop('quant_bits')
            if 'input' in quant_types:
                target_names.append(INPUT_PREFIX)
                quant_bit = quant_bits if isinstance(quant_bits, int) else quant_bits['input']
                target_settings[INPUT_PREFIX] = {'quant_dtype': f'int{quant_bit}'}
            if 'weight' in quant_types:
                target_names.extend(['weight', 'bias'])
                quant_bit = quant_bits if isinstance(quant_bits, int) else quant_bits['weight']
                target_settings['weight'] = {'quant_dtype': f'int{quant_bit}'}
                target_settings['bias'] = {'quant_dtype': f'int{quant_bit}'}
            if 'output' in quant_types:
                target_names.append(OUTPUT_PREFIX)
                quant_bit = quant_bits if isinstance(quant_bits, int) else quant_bits['output']
                target_settings[OUTPUT_PREFIX] = {'quant_dtype': f'int{quant_bit}'}
            config['target_names'] = target_names
            config['target_settings'] = target_settings
    return config_list


def select_modules_by_config(model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[Dict[str, torch.nn.Module], Dict[str, Any]]:
    # return ({module_name: module}, public_config)
    # intersection(union(op_names, op_names_re), op_types) - exclude_op_names - exclude_op_names_re - exclude_op_types
    name2module = {}
    type2names = defaultdict(set)
    for module_name, module in model.named_modules(remove_duplicate=False):
        name2module[module_name] = module
        type2names[type(module).__name__].add(module_name)

    config = deepcopy(config)
    op_names = set(config.pop('op_names', list()))
    op_types = config.pop('op_types', list())
    op_names_re = config.pop('op_names_re', list())
    exclude_op_names = config.pop('exclude_op_names', list())
    exclude_op_types = config.pop('exclude_op_types', list())
    exclude_op_names_re = config.pop('exclude_op_names_re', list())

    for op_name_re in op_names_re:
        for op_name in name2module:
            if re.match(op_name_re, op_name):
                op_names.add(op_name)

    if op_types:
        selected_by_op_types = set()
        for op_type in op_types:
            selected_by_op_types.update(type2names.get(op_type, set()))
        op_names.intersection_update(selected_by_op_types)

    op_names.difference_update(exclude_op_names)

    for op_name_re in exclude_op_names_re:
        for op_name in name2module:
            if re.match(op_name_re, op_name) and op_name in op_names:
                op_names.remove(op_name)

    for op_type in exclude_op_types:
        op_names.difference_update(type2names.get(op_type, set()))

    return {module_name: name2module[module_name] for module_name in op_names}, config
