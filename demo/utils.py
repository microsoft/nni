from copy import deepcopy
from typing import Dict, List

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
