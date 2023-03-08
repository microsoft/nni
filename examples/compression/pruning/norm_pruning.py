from nni.contrib.compression.pruning import (
    L1NormPruner,
    L2NormPruner,
    FPGMPruner
)
from nni.contrib.compression.utils import auto_set_denpendency_group_ids
from nni.compression.pytorch.speedup.v2 import ModelSpeedup

import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_small

if __name__ == '__main__':
    model = mobilenet_v3_small()
    config_list = [{
        'op_types': ['Conv2d'],
        'op_names_re': ['features.*'],
        'sparse_ratio': 0.5
    }]
    dummy_input = torch.rand(8, 3, 224, 224)
    config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
    pruner = L1NormPruner(model, config_list)
    _, masks = pruner.compress()
    pruner.unwrap_model()

    model = ModelSpeedup(model, dummy_input, masks).speedup_model()

    model(dummy_input)
