# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from nni.compression.pruning import TaylorPruner, LinearPruner, AGPPruner
from nni.compression.utils import auto_set_denpendency_group_ids
from nni.compression.speedup import ModelSpeedup

schedule_type = 'agp'


if __name__ == '__main__':
    # finetuning resnet18 on Cifar10
    model = build_resnet18()
    optimizer = prepare_optimizer(model)
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=10)
    _, test_loader = prepare_dataloader()
    print('Original model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Original model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')

    config_list = [{
        'op_types': ['Conv2d'],
        'sparse_ratio': 0.5
    }]
    dummy_input = torch.rand(8, 3, 224, 224).to(device)
    config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
    optimizer = prepare_optimizer(model)
    evaluator = TorchEvaluator(train, optimizer, training_step)

    sub_pruner = TaylorPruner(model, config_list, evaluator, training_steps=100)
    if schedule_type == 'agp':
        scheduled_pruner = AGPPruner(sub_pruner, interval_steps=100, total_times=10)
    elif schedule_type == 'linear':
        scheduled_pruner = LinearPruner(sub_pruner, interval_steps=100, total_times=10)

    _, masks = scheduled_pruner.compress(max_steps=100 * 10, max_epochs=None)
    scheduled_pruner.unwrap_model()

    model = ModelSpeedup(model, dummy_input, masks).speedup_model()
    print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Pruned model without finetuning acc: ', evaluate(model, test_loader), '%')

    optimizer = prepare_optimizer(model)
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=10)
    _, test_loader = prepare_dataloader()
    print('Pruned model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')
