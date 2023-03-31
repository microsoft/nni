# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from ..registry import model_zoo

def dummy_inputs(*_):
    return {'ims': torch.randn(1, 3, 640, 640)}

def config_list(*_):
    return [{
        'sparsity': 0.2,
        'op_types': ['Conv2d'],
    }]

def yolov5s():
    mod = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return mod

model_zoo.register(
    'yolo', 'yolov5',
    yolov5s,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    need_run=True,
)
