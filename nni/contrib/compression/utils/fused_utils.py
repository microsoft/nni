# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Any, Set

import torch
import torch.nn as nn

from ..base.setting import OUTPUT_PREFIX


FUSED_MODULES_TYPES_LIST = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, \
        nn.ConvTranspose2d, nn.ConvTranspose3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

def get_module(model: torch.nn.Module, module_name: str):
    module_name_list = module_name.strip().split('.')
    cur_module = model
    for s in module_name_list:
        cur_module = getattr(cur_module, s, None)
        if cur_module is None:
            raise ValueError(f"can\'t find {module_name} in the model")

    return cur_module


def validate_fused_modules_config(fuse_module_names: List[str], model: nn.Module, config: Dict[str, Any]):
    # check the type and length
    if not isinstance(fuse_module_names, List) or not all(isinstance(_, str) for _ in fuse_module_names):
        raise ValueError(f"Expected fused_modules type is List[List[str]], but got {type(fuse_module_names)}")
    assert len(fuse_module_names) >=2 and len(fuse_module_names) <= 3, \
            f"Only support fuse two or three modules, but got {len(fuse_module_names)} modules"
    name2module = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        name2module[module_name] = module
    # check the validation of operations
    for i, module_name in enumerate(fuse_module_names):
        assert module_name in name2module, \
            f"The module fusion configuration of {module_name} should be defined in the config_list"
        module = name2module[module_name]
        if i == 0 and type(module) not in FUSED_MODULES_TYPES_LIST:
            raise ValueError(f"{module_name} is not supported to fuse, please register it in FUSED_MODULES_TYPES_LIST")
        if i != 0 and type(module) == torch.nn.ReLU:
            assert OUTPUT_PREFIX in config.get('target_names', []), \
                "Please provide the quant setting of the output in the config_list"


def get_fused_module_list(module_name: str, mode: str, fused_module_names: List[str]):
    if mode != 'quantization' and len(fused_module_names) > 0:
        raise ValueError("module fusion is only supported for quantization, please check it")
    elif mode != 'quantization' or len(fused_module_names) == 0:
        return []
    if module_name == fused_module_names[0]:
        return fused_module_names
    else:
        raise ValueError(f"can't find fusion configuration for {module_name}")


def update_config(wrapper_config: Dict[str, Dict[str, Any]], configs: Dict[str, Dict[str, Any]]):
    # sourcery skip: merge-duplicate-blocks, remove-redundant-if
    wrapper_config = wrapper_config if wrapper_config else {}
    for mode, config in configs.items():
        if mode not in wrapper_config:
            wrapper_config[mode] = config
        else:
            for name, value in config.items():
                if name not in wrapper_config[mode]:
                    wrapper_config[mode][name] = value
                elif isinstance(value, (Dict, Set)):
                    wrapper_config[mode][name].update(value)
                elif isinstance(value, List):
                    wrapper_config[mode][name] = list(set(wrapper_config[mode][name] + value))
                else:
                    wrapper_config[mode][name] = value

    return wrapper_config


def check_bias(module):
    try:
        return isinstance(module.bias.data, torch.Tensor)
    except AttributeError:
        return False
