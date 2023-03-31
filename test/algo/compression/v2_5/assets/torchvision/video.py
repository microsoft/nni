# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torchvision.models.video as tm_vid

from ..registry import model_zoo


def config_list(*_):
    return [{
        'sparsity': 0.2,
        'op_types': ['Conv2d']
    }]

def dummy_inputs(*_):
    return {'x': torch.randn(2, 3, 16, 224, 224)}

model_zoo.register(
    'torchvision.video', 'mc3_18',
    tm_vid.mc3_18,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (empty mask)',
)

model_zoo.register(
    'torchvision.video', 'mvit_v2_s',
    tm_vid.mvit_v2_s,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)

model_zoo.register(
    'torchvision.video', 'r2plus1d_18',
    tm_vid.r2plus1d_18,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (empty mask)',
)

model_zoo.register(
    'torchvision.video', 'r3d_18',
    tm_vid.r3d_18,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (empty mask)',
)

model_zoo.register(
    'torchvision.video', 's3d',
    tm_vid.s3d,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (empty mask)',
)
