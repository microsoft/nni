# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Any, Literal
from collections import defaultdict

import torch
import torch.nn as nn

from ..base.setting import OUTPUT_PREFIX
from ..base.config import select_modules_by_config

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


def validate_fused_modules_config(model: nn.Module, config_list: List[Dict[str, Any]], \
                        fused_modules_names_lis: List[List[str]]):
    # check the type of fused_modules_names_lis
    if not isinstance(fused_modules_names_lis, List) or not all(isinstance(_, List) for _ in fused_modules_names_lis):
        raise ValueError(f"Expected fused_modules type is List[List[str]], but got {type(fused_modules_names_lis)}")

    modulename2config = defaultdict(dict)
    for config in config_list:
        modules, public_config = select_modules_by_config(model, config)
        for module_name, module in modules.items():
            modulename2config[module_name].update(public_config)
    # check the validation of operations
    for fused_modules_names in fused_modules_names_lis:
        assert fused_modules_names is not None and len(fused_modules_names) >=2 and len(fused_modules_names) <= 3, \
            f"Only support fusion two modules or three modules, but got {len(fused_modules_names)} modules"
        for i, module_name in enumerate(fused_modules_names):
            if i == 0:
                module = get_module(model, module_name) # check whtether the module is in the model
                assert module_name in modulename2config, \
                    f"The quantization configuration of {module_name} should be defined in the config_list"
                continue
            module = get_module(model, module_name)
            if module_name in modulename2config:
                raise ValueError(f"Don't provide quantization configuration for the fused module: {module_name}")

            fused_module_config = modulename2config[fused_modules_names[0]]

            if type(module) == torch.nn.ReLU:
                assert OUTPUT_PREFIX in fused_module_config.get("target_names", []), \
                    "Please provide the quant setting of the output in the config_list"


def find_fused_module_list(model: nn.Module, fused_modules_names_lis: List[List[str]], \
         module_name: str, mode: Literal['pruning', 'quantization', 'distillation']):
    '''
    Note that , the first element is used for the main module, others will be replaced by identity func
    '''
    # layer fusion only supports quantization
    if mode != "quantization":
        return []

    sub_module = get_module(model, module_name)
    if type(sub_module) not in FUSED_MODULES_TYPES_LIST:
        return []

    for fused_module_names in fused_modules_names_lis:
        if module_name == fused_module_names[0]:
            return fused_module_names

    return []


def get_identity_module_set(fused_modules_names_lis: List[List[str]]):
    identity_modules_set = set()
    for fused_modules_names in fused_modules_names_lis:
         identity_modules_set.update(set(fused_modules_names[1:]))
    return identity_modules_set


def update_config(wrapper_config: Dict[str, Dict[str, Any]], configs: Dict[str, Dict[str, Any]]):
    wrapper_config = wrapper_config if wrapper_config else {}

    for name, config in configs.items():
        if name not in wrapper_config:
            wrapper_config[name] = config
        else:
            raise RuntimeError(f"Can't use {name} method to compress the same module twice")

    return wrapper_config


def check_bias(module):
    try:
        return isinstance(module.bias.data, torch.Tensor)
    except AttributeError:
        return False
