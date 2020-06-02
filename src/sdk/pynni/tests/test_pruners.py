# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from unittest import TestCase, main
from nni.compression.torch import LevelPruner, SlimPruner, FPGMPruner, L1FilterPruner, \
    L2FilterPruner, AGP_Pruner, ActivationMeanRankFilterPruner, ActivationAPoZRankFilterPruner, \
    TaylorFOWeightFilterPruner

def validate_sparsity(wrapper, sparsity, bias=False):
    masks = [wrapper.weight_mask]
    if bias and wrapper.bias_mask is not None:
        masks.append(wrapper.bias_mask)
    for m in masks:
        actual_sparsity = (m == 0).sum().item() / m.numel()
        msg = 'actual sparsity: {:.2f}, target sparsity: {:.2f}'.format(actual_sparsity, sparsity)
        assert math.isclose(actual_sparsity, sparsity, abs_tol=0.1), msg

prune_config = {
    'level': {
        'pruner_class': LevelPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['default'],
        }],
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, False),
            lambda model: validate_sparsity(model.fc, 0.5, False)
        ]
    },
    'agp': {
        'pruner_class': AGP_Pruner,
        'config_list': [{
            'initial_sparsity': 0.,
            'final_sparsity': 0.8,
            'start_epoch': 0,
            'end_epoch': 10,
            'frequency': 1,
            'op_types': ['default']
        }],
        'validators': []
    },
    'slim': {
        'pruner_class': SlimPruner,
        'config_list': [{
            'sparsity': 0.7,
            'op_types': ['BatchNorm2d']
        }],
        'validators': [
            lambda model: validate_sparsity(model.bn1, 0.7, model.bias)
        ]
    },
    'fpgm': {
        'pruner_class': FPGMPruner,
        'config_list':[{
            'sparsity': 0.5,
            'op_types': ['Conv2d']
        }],
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, model.bias)
        ]
    },
    'l1': {
        'pruner_class': L1FilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
        }],
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, model.bias)
        ]
    },
    'l2': {
        'pruner_class': L2FilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
        }],
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, model.bias)
        ]
    },
    'taylorfo': {
        'pruner_class': TaylorFOWeightFilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
        }],
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, model.bias)
        ]
    },
    'mean_activation': {
        'pruner_class': ActivationMeanRankFilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
        }],
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, model.bias)
        ]
    },
    'apoz': {
        'pruner_class': ActivationAPoZRankFilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
        }],
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, model.bias)
        ]
    }
}

class Model(nn.Module):
    def __init__(self, bias=True):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2, bias=bias)
        self.bias = bias
    def forward(self, x):
        return self.fc(self.pool(self.bn1(self.conv1(x))).view(x.size(0), -1))

def pruners_test(pruner_names=['level', 'slim', 'fpgm', 'l1', 'l2', 'taylorfo', 'mean_activation', 'apoz'], bias=True):
#def pruners_test(pruner_names=['l1'], bias=True):
    for pruner_name in pruner_names:
        model = Model(bias=bias)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config_list = prune_config[pruner_name]['config_list']

        x = torch.randn(2, 1, 28, 28)
        y = torch.tensor([0, 1]).long()
        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pruner = prune_config[pruner_name]['pruner_class'](model, config_list, optimizer)
        pruner.compress()

        x = torch.randn(2, 1, 28, 28)
        y = torch.tensor([0, 1]).long()
        out = model(x)
        print(dir(model.conv1))
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pruner_name == 'taylorfo':
            # taylorfo algorithm calculate contributions at first iteration(step), and do pruning
            # when iteration >= statistics_batch_num (default 1)
            optimizer.step()

        pruner.export_model('./model_tmp.pth', './mask_tmp.pth', './onnx_tmp.pth', input_shape=(2,1,28,28))

        for v in prune_config[pruner_name]['validators']:
            v(model)

    os.remove('./model_tmp.pth')
    os.remove('./mask_tmp.pth')
    os.remove('./onnx_tmp.pth')

class PrunerTestCase(TestCase):
    def test_pruners(self):
        pruners_test(bias=True)

    def test_pruners_no_bias(self):
        pruners_test(bias=False)

if __name__ == '__main__':
    main()
