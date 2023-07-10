# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module


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
            total_weight_num += module.weight.data.numel()  # type: ignore
        for module_name, module in compact_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            left_weight_num += module.weight.data.numel()  # type: ignore
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
            module_weight_num = module.weight.data.numel()  # type: ignore
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
    and it may have different structure with origin_model cause of speedup.

    Parameters
    ----------
    origin_model : torch.nn.Module
        The original un-pruned model.
    compact_model : torch.nn.Module
        The model after speedup or original model.
    compact_model_masks: Dict[str, Dict[str, Tensor]]
        The masks applied on the compact model, if the original model have been speedup, this should be {}.
    config_list : List[Dict]
        The config_list used by pruning the original model.

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


def get_model_weights_numel(model: Module, config_list: List[Dict],
                            masks: Dict[str, Dict[str, Tensor]] = {}) -> Tuple[Dict[str, int], Dict[str, float]]:
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
                model_weights_numel[module_name] = module.weight.data.numel()  # type: ignore
    return model_weights_numel, masked_rate


def get_output_batch_dims(t: Tensor, module: Module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Bilinear)):
        batch_nums = math.prod(t.shape[:-1])
        batch_dims = [_ for _ in range(len(t.shape[:-1]))]
    elif isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        batch_nums = math.prod(t.shape[:-2])
        batch_dims = [_ for _ in range(len(t.shape[:-2]))]
    elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        batch_nums = math.prod(t.shape[:-3])
        batch_dims = [_ for _ in range(len(t.shape[:-3]))]
    elif isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        batch_nums = math.prod(t.shape[:-4])
        batch_dims = [_ for _ in range(len(t.shape[:-4]))]
    else:
        raise TypeError(f'Found unsupported module type in activation based pruner: {module.__class__.__name__}')
    return batch_dims, batch_nums
