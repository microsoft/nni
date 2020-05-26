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
    NetAdaptPruner, SimulatedAnnealingPruner, ADMMPruner, AutoCompressPruner

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
    },
    'netadapt': {
        'pruner_class': NetAdaptPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d']
        }],
        'fine_tuner': lambda model:model, 
        'evaluator':lambda model: 0.9,
        'validators': []
    },
    'simulatedannealing': {
        'pruner_class': SimulatedAnnealingPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d']
        }],
        'evaluator':lambda model: 0.9,
        'validators': []
    },
    'admm': {
        'pruner_class': ADMMPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
        }],
        'trainer': lambda model, optimizer, criterion, epoch, callback : model, 
        'validators': [
            lambda model: validate_sparsity(model.conv1, 0.5, model.bias)
        ]
    },
    'autocompress': {
        'pruner_class': AutoCompressPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
        }],
        'trainer': lambda model, optimizer, criterion, epoch, callback : model,
        'evaluator': lambda model: 0.9,
        'dummy_input': torch.randn([64, 1, 28, 28]),
        'validators': []
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

def pruners_test(pruner_names=['level', 'agp', 'slim', 'fpgm', 'l1', 'l2', 'mean_activation', 'apoz', 'netadapt', 'simulatedannealing', 'admm', 'autocompress'], bias=True):
    for pruner_name in pruner_names:
        print('testing {}...'.format(pruner_name))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(bias=bias).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config_list = prune_config[pruner_name]['config_list']

        x = torch.randn(2, 1, 28, 28).to(device)
        y = torch.tensor([0, 1]).long().to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pruner_name == 'netadapt':
            pruner = prune_config[pruner_name]['pruner_class'](model, config_list, fine_tuner=prune_config[pruner_name]['fine_tuner'], evaluator=prune_config[pruner_name]['evaluator'])
        elif pruner_name == 'simulatedannealing':
            pruner = prune_config[pruner_name]['pruner_class'](model, config_list, evaluator=prune_config[pruner_name]['evaluator'])
        elif pruner_name == 'admm':
            pruner = prune_config[pruner_name]['pruner_class'](model, config_list, trainer=prune_config[pruner_name]['trainer'])
        elif pruner_name == 'autocompress':
            pruner = prune_config[pruner_name]['pruner_class'](model, config_list, trainer=prune_config[pruner_name]['trainer'], evaluator=prune_config[pruner_name]['evaluator'], dummy_input=x)
        else:
            pruner = prune_config[pruner_name]['pruner_class'](model, config_list, optimizer)
        pruner.compress()

        x = torch.randn(2, 1, 28, 28).to(device)
        y = torch.tensor([0, 1]).long().to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pruner.export_model('./model_tmp.pth', './mask_tmp.pth', './onnx_tmp.pth', input_shape=(2,1,28,28), device=device)

        for v in prune_config[pruner_name]['validators']:
            v(model)

    
    filePaths = ['./model_tmp.pth', './mask_tmp.pth', './onnx_tmp.pth', './search_history.csv', './search_result.json']
    for f in filePaths:
        if os.path.exists(f):
            os.remove(f)

class PrunerTestCase(TestCase):
    def test_pruners(self):
        pruners_test(bias=True)

    def test_pruners_no_bias(self):
        pruners_test(bias=False)

if __name__ == '__main__':
    main()
