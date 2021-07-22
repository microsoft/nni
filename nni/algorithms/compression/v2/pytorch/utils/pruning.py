# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Module


def apply_compression_results(model: Module, masks: Tensor):
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
        real_config_list[-1]['sparsity'] = 1 - left_weight_num / total_weight_num
    return real_config_list


def compute_sparsity_with_mask(masked_model: Module, masks: Dict[str, Tensor], config_list: List[Dict], dim: int = 0):
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
            weight_mask = masks[module_name]['weight']
            mask_size = weight_mask.size()
            if len(mask_size) == 1:
                index = torch.nonzero(weight_mask.abs() != 0).tolist()
            else:
                sum_idx = list(range(len(mask_size)))
                sum_idx.remove(dim)
                index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0).tolist()
            module_weight_num = module.weight.data.numel()
            left_weight_num += module_weight_num * len(index) / weight_mask.size(dim)
            total_weight_num += module_weight_num
        real_config_list.append(deepcopy(config))
        real_config_list[-1]['sparsity'] = 1 - left_weight_num / total_weight_num
    return real_config_list
