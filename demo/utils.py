from copy import deepcopy
from typing import Dict, List

import torch
from torch.nn import Module


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
    exclude = []
    exclude_idxes = []
    config_list = deepcopy(config_list)
    for idx, config in reversed(list(enumerate(config_list))):
        if 'exclude' in config:
            exclude.extend(config['op_names'])
            exclude_idxes.append(idx)
            continue
        config['op_names'] = sorted(list(set(config['op_names']).difference(set(exclude))))
        exclude.extend(config['op_names'])
    for idx in sorted(exclude_idxes, reverse=True):
        config_list.pop(idx)
    return config_list

def get_model_weight_numel(model: Module, config_list: List[Dict]) -> Dict:
    model_weight = {}
    for config in config_list:
        for module_name, module in model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            model_weight[module_name] = module.weight.data.numel()
    return model_weight

def compute_sparsity_with_compact_model(origin_model: Module, compact_model, config_list: List[Dict]) -> List[Dict]:
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
        real_config_list.append(deepcopy(config_list))
        real_config_list[-1]['sparsity'] = 1 - left_weight_num / total_weight_num
    return real_config_list

def compute_sparsity_with_mask(model: Module, masks_file: str, config_list: List[Dict], dim: int = 0):
    masks = torch.load(masks_file)
    real_config_list = []
    for config in config_list:
        left_weight_num = 0
        total_weight_num = 0
        for module_name, module in model.named_modules():
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
        real_config_list.append(deepcopy(config_list))
        real_config_list[-1]['sparsity'] = 1 - left_weight_num / total_weight_num
    return real_config_list


if __name__ == '__main__':
    from examples.model_compress.models.cifar10.vgg import VGG
    model = VGG()
    config_list = [{
        'op_types': ['Conv2d']
    }, {
        'exclude': '',
        'op_names': ['feature.3', 'feature.7']
    }, {
        'exclude': '',
        'op_names': ['feature.0', 'feature.10']
    }]
    print(dedupe_config_list(unfold_config_list(model, config_list)))
    print(get_model_weight_numel(model, config_list))
