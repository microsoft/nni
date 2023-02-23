# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import sys
from copy import deepcopy

import torch
import torch.nn.functional as F

sys.path.append("./nni/")

from nni.contrib.compression.base.config import trans_legacy_config_list
from nni.contrib.compression.base.wrapper import register_wrappers

torch.manual_seed(1024)
device = 'cuda'

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)
        self.batchnorm1 = torch.nn.BatchNorm2d(20)

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, x.size()[1:].numel())
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = NaiveModel().to(device)

config_list_1 = [
    {
    'op_names': ['fc1', 'fc2'],
    'target_names': ['_input_', 'weight', '_output_'],
    'quant_dtype': 'int8',
    'quant_scheme': 'affine',
    'granularity': 'default'
},
{
    'op_names': ['conv2', "conv1"],
    'target_names': ['_output_','weight', '_input_'],
    'quant_dtype': 'int2',
    'quant_scheme': 'affine',
    'granularity': 'default',
},
{
    'op_names': ['relu2', 'relu1'],
    'target_names': ['_output_'],
    'quant_dtype': 'int4',
    'quant_scheme': 'affine',
    'granularity': 'default',
}]

config_list_2 = [{
    'op_names': ['conv1'],
    'target_names': ['weight', '_input_', "_output_", "bias"],
    'quant_dtype': 'int2',
    'quant_scheme': 'affine',
    'granularity': 'default',
    'fuse_names': ["conv1", "batchnorm1"]
}]


config_list_1 = trans_legacy_config_list(deepcopy(config_list_1))
config_list_2 = trans_legacy_config_list(deepcopy(config_list_2))
print(config_list_1)
print(config_list_2)

module_wrappers_1, target_spaces_1 = register_wrappers(model, config_list_1, "quantization")
print(f"target_space_1={target_spaces_1}\n")
for module_name, wrapper in module_wrappers_1.items():
    print(f"module_name={module_name}\tconfig={wrapper.config}\twrapper={wrapper}\n")

module_wrappers_2, target_spaces_2 = register_wrappers(model, config_list_2, "quantization", module_wrappers_1)
print(f"target_space_2={target_spaces_2}\n")
for module_name, wrapper in module_wrappers_2.items():
    print(f"module_name={module_name}\tconfig={wrapper.config}\twrapper={wrapper}\n")
print(f"target_space_2={target_spaces_2}\n")
