# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torchvision.models.detection as tm_det

from ..registry import model_zoo


def config_list(*_):
    return [{
        'sparsity': 0.2,
        'op_types': ['Conv2d']
    }]

model_zoo.register(
    'torchvision.detection', 'fasterrcnn',
    tm_det.fasterrcnn_resnet50_fpn, # (images, targets)
    dummy_inputs=lambda self: {
        'images': [torch.randn(3, 224, 224)],
        'targets': [
            {
                'boxes': torch.tensor([[10, 50, 40, 60]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
        }], 
    },
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)

model_zoo.register(
    'torchvision.detection', 'fcos',
    tm_det.fcos_resnet50_fpn,   # (images, targets)
    lambda self: {
        'images': [torch.randn(3, 224, 224)],
        'targets': [
            {
                'boxes': torch.tensor([[10, 50, 40, 60]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
        }],
    },
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)

model_zoo.register(
    'torchvision.detection', 'maskrcnn',
    tm_det.maskrcnn_resnet50_fpn,   # (images, targets)
    dummy_inputs=lambda self: {
        'images': [torch.randn(3, 224, 224)],
        'targets': [
            {
                'boxes': torch.tensor([[10, 50, 40, 60]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'masks': torch.zeros(1, 224, 224, dtype=torch.uint8),
        }],
    },
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)

model_zoo.register(
    'torchvision.detection', 'retinanet',
    tm_det.retinanet_resnet50_fpn,    # (images, targets)
    dummy_inputs=lambda self: {
        'images': [torch.randn(3, 224, 224)],
        'targets': [
            {
                'boxes': torch.tensor([[10, 50, 40, 60]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
        }],
    },
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)

model_zoo.register(
    'torchvision.detection', 'ssd',
    tm_det.ssdlite320_mobilenet_v3_large, 
    dummy_inputs=lambda self: {
        'images': [torch.randn(3, 320, 320), torch.randn(3, 320, 320)],
        'targets': [
            {
                'boxes': torch.tensor([[10, 50, 40, 60]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
            },
            {
                'boxes': torch.tensor([[10, 50, 40, 60]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
            }
            ],
    },
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)


model_zoo.register(
    'torchvision.detection', 'keypointrcnn',
    tm_det.keypointrcnn_resnet50_fpn,   # (images, targets)
    dummy_inputs=lambda self: {
        'images': [torch.randn(3, 224, 224)],
        'targets': [
            {
                'boxes': torch.tensor([[10, 50, 40, 60]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'keypoints': torch.tensor([[10, 50, 0], [20, 60, 1]], dtype=torch.float32),
            }],
    },
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)