# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

######################################################################################
# NOTE: copy from branch wrapper-refactor, will rm this file in this or next release.#
######################################################################################

from copy import deepcopy
import logging
from typing import Any, Dict, List

from torch.nn import Module

_logger = logging.getLogger(__name__)


def _unfold_op_partial_names(model: Module, config_list: List[Dict]) -> List[Dict]:
    config_list = deepcopy(config_list)
    full_op_names = [op_name for op_name, _ in model.named_modules()]
    for config in config_list:
        op_names = config.pop('op_names', [])
        op_partial_names = config.pop('op_partial_names', [])
        for op_partial_name in op_partial_names:
            op_names.extend([op_name for op_name in full_op_names if op_partial_name in op_name])
        config['op_names'] = list(set(op_names))
    return config_list


def unfold_config_list(model: Module, config_list: List[Dict]) -> Dict[str, Dict[str, Any]]:
    '''
    Unfold config_list to op_names level, return a config_dict {op_name: config}.
    '''
    config_list = _unfold_op_partial_names(model=model, config_list=config_list)
    config_dict = {}
    for config in config_list:
        for key in ['op_types', 'op_names', 'exclude_op_names']:
            config.setdefault(key, [])
        op_names = []
        for module_name, module in model.named_modules():
            module_type = type(module).__name__
            if (module_type in config['op_types'] or module_name in config['op_names']) and module_name not in config['exclude_op_names']:
                op_names.append(module_name)
        config_template = deepcopy(config)
        for key in ['op_types', 'op_names', 'exclude_op_names']:
            config_template.pop(key, [])
        for op_name in op_names:
            if op_name in config_dict:
                warn_msg = f'{op_name} duplicate definition of config, replace old config:\n' + \
                           f'{config_dict[op_name]}\n' + \
                           f'with new config:\n{config_template}\n'
                _logger.warning(warn_msg)
            config_dict[op_name] = deepcopy(config_template)
    return config_dict
