# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torchvision.models as tm

from ..registry import model_zoo


def config_list(*_):
    return [{
        'sparsity': 0.2,
        'op_types': ['Conv2d']
    }]
    
def dummy_inputs(*_):
    return {'x': torch.randn(2, 3, 224, 224)}

model_zoo.register(
    'torchvision', 'alexnet',
    tm.alexnet, 
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'convnext',
    tm.convnext_tiny,
    dummy_inputs=dummy_inputs,
    config_list=lambda self: [{
        'sparse_ratio': 0.5,
        'op_types': ['Conv2d', 'Linear'],
        'op_names_re': ['features.*'],
    }],
    need_auto_set_dependency=True,
    skip_reason='cannot prune (empty layer)',
)

model_zoo.register(
    'torchvision', 'densenet',
    tm.densenet121,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'efficientnet',
    tm.efficientnet_v2_s,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'googlenet',
    tm.googlenet,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='output is dict',
)

model_zoo.register(
    'torchvision', 'inception',
    tm.inception_v3,
    dummy_inputs=lambda self: {'x': torch.randn(2, 3, 299, 299)},
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'mobilenet',
    tm.mobilenet_v2,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'maxvit',
    tm.maxvit_t,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'mnasnet',
    tm.mnasnet0_5,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'resnet',
    tm.resnet18,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'regnet',
    tm.regnet_x_16gf,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'resnext',
    tm.resnext50_32x4d,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'shufflenet',
    tm.shufflenet_v2_x0_5,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (endless loop)',
)

model_zoo.register(
    'torchvision', 'squeezenet',
    tm.squeezenet1_0,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'swin',
    tm.swin_s,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'vgg',
    tm.vgg11,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'torchvision', 'vit',
    tm.vit_b_16,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (jit trace error)',
)

model_zoo.register(
    'torchvision', 'wide_resnet',
    tm.wide_resnet50_2,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)