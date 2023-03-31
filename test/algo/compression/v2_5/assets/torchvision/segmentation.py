# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torchvision.models.segmentation as tm_seg

from ..registry import model_zoo


def config_list(*_):
    return [{
        'sparsity': 0.2,
        'op_types': ['Conv2d']
    }]
    
def dummy_inputs(*_):
    return {'x': torch.randn(2, 3, 224, 224)}

model_zoo.register(
    'torchvision.segmentation', 'deeplabv3',
    tm_seg.deeplabv3_resnet50,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision.segmentation', 'fcn',
    tm_seg.fcn_resnet50,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision.segmentation', 'lraspp',
    tm_seg.lraspp_mobilenet_v3_large,             
    dummy_inputs=lambda self: {'input': torch.randn(2, 3, 224, 224)},
    config_list=config_list,
    skip_reason='cannot prune (fix mask conflict)',
)
