# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn

from ..base.setting import OUTPUT_PREFIX


FUSED_MODULES_TYPES_LIST = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, \
        nn.ConvTranspose2d, nn.ConvTranspose3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
ACTIVATION_MODULES_TYPE_LIST = [torch.nn.ReLU]


def validate_fused_modules_config(fuse_module_names: List[Tuple[str]], model: nn.Module, config: Dict[str, Any]):
    # check the type and length
    if not isinstance(fuse_module_names, List) or not all(isinstance(_, tuple) for _ in fuse_module_names) \
        or not all(isinstance(_, str) for fused_pair in fuse_module_names for _ in fused_pair):
        raise ValueError(f"Expected the type of fused_modules is List[Tuple[str]], but got {type(fuse_module_names)}")
    name2module = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        name2module[module_name] = module
    # check the validation of operations
    for fused_pair in fuse_module_names:
        assert len(fused_pair) >=2 and len(fused_pair) <= 3, \
            f"Only 2 or 3 modules are supported for fusion, but got {len(fused_pair)}"
        for i, module_name in enumerate(fused_pair):
            assert module_name in name2module, \
                f"{module_name} doesn\'t exist in the model"
            module = name2module[module_name]
            if i == 0 and type(module) not in FUSED_MODULES_TYPES_LIST:
                raise ValueError(f"{module_name} is not supported for module fusion, \
                                 please register it in the FUSED_MODULES_TYPES_LIST")
            if i != 0 and type(module) in ACTIVATION_MODULES_TYPE_LIST:
                assert OUTPUT_PREFIX in config.get('target_names', []), \
                    "If you need to fuse activation functions, a quantization setting for the output in the config_list should be provided"


def get_fused_module_list(module_name: str, mode: str, fused_module_names: List[Tuple[str]]) -> Tuple:
    if mode != 'quantization' and len(fused_module_names) > 0:
        raise ValueError(f"Only quantization supports model fusion, but got {mode}")
    elif len(fused_module_names) == 0:
        return ()
    for fuse_pair in fused_module_names:
        if module_name == fuse_pair[0]:
            return fuse_pair

    return ()


def update_config(wrapper_config: Dict[str, List[Dict[str, Any]]], configs: Dict[str, Dict[str, Any]]):
    # sourcery skip: merge-duplicate-blocks, remove-redundant-if
    for key, config in configs.items():
        if key not in wrapper_config:
            wrapper_config[key] = []
        wrapper_config[key].append(config)

    return wrapper_config


def check_bias(module):
    if getattr(module, 'bias', 'non-exist') == 'non-exist':
        return 'non-exist'
    try:
        return 'Tensor' if isinstance(module.bias.data, torch.Tensor) else 'None'
    except AttributeError:
        return 'None'
