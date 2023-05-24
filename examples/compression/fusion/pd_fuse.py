# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is an exmaple for how to fuse pruning and distillation.
"""

import pickle

import torch

from examples.compression.models import (
    build_resnet18,
    prepare_dataloader,
    prepare_optimizer,
    train,
    training_step,
    evaluate,
    device
)

from nni.compression import TorchEvaluator
from nni.compression.distillation import DynamicLayerwiseDistiller
from nni.compression.pruning import TaylorPruner, AGPPruner
from nni.compression.utils import auto_set_denpendency_group_ids
from nni.compression.speedup import ModelSpeedup


if __name__ == '__main__':
    # finetuning resnet18 on Cifar10
    model = build_resnet18()
    optimizer = prepare_optimizer(model)
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=30)
    _, test_loader = prepare_dataloader()
    print('Original model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Original model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')

    # build a teacher model
    teacher_model = build_resnet18()
    teacher_model.load_state_dict(pickle.loads(pickle.dumps(model.state_dict())))

    # create pruner
    bn_list = [module_name for module_name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
    config_list = [{
        'op_types': ['Conv2d'],
        'sparse_ratio': 0.5
    }, *[{
        'op_names': [name],
        'target_names': ['_output_'],
        'target_settings': {
            '_output_': {
                'align': {
                    'module_name': name.replace('bn', 'conv') if 'bn' in name else name.replace('downsample.1', 'downsample.0'),
                    'target_name': 'weight',
                    'dims': [0],
                },
                'granularity': 'per_channel'
            }
        }
    } for name in bn_list]]
    dummy_input = torch.rand(8, 3, 224, 224).to(device)
    config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)

    optimizer = prepare_optimizer(model)
    evaluator = TorchEvaluator(train, optimizer, training_step)
    sub_pruner = TaylorPruner(model, config_list, evaluator, training_steps=100)
    scheduled_pruner = AGPPruner(sub_pruner, interval_steps=100, total_times=30)

    # create distiller
    def teacher_predict(batch, teacher_model):
        return teacher_model(batch[0])

    config_list = [{
        'op_types': ['Conv2d'],
        'lambda': 0.1,
        'apply_method': 'mse',
    }]
    distiller = DynamicLayerwiseDistiller.from_compressor(scheduled_pruner, config_list, teacher_model, teacher_predict, 0.1)

    # max_steps contains (30 iterations 100 steps agp taylor pruning, and 3000 steps finetuning)
    distiller.compress(max_steps=100 * 60, max_epochs=None)
    distiller.unwrap_model()
    distiller.unwrap_teacher_model()

    # speed up model
    masks = scheduled_pruner.get_masks()
    model = ModelSpeedup(model, dummy_input, masks).speedup_model()
    print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Pruned model without finetuning acc: ', evaluate(model, test_loader), '%')
