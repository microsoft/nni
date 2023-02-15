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

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, x.size()[1:].numel())
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = NaiveModel().to(device)
print(model)

config_list = [
    {
    'op_names': ['fc1', 'fc2'],
    'target_names': ['_input_', 'weight', '_output_'],
    'quant_dtype': 'int8',
    'quant_scheme': 'affine',
    'granularity': 'default',
},
{
    'op_names': ['conv1', 'conv2'],
    'target_names': ['_output_','weight', '_input_'],
    'quant_dtype': 'int2',
    'quant_scheme': 'affine',
    'granularity': 'default',
},
{
    'op_names': ['relu2'],
    'target_names': ['_output_'],
    'quant_dtype': 'int4',
    'quant_scheme': 'affine',
    'granularity': 'default',
}]


fused_modules = [["fc1","relu3"], ["conv1", "relu1"]]

config_list = trans_legacy_config_list(deepcopy(config_list))
print(config_list)

module_wrappers, target_spaces = register_wrappers(model, config_list, "quantization", fused_modules_names_lis=fused_modules)
print(f"target_space_1={target_spaces}\n")
for module_name, wrapper in module_wrappers.items():
    print(f"module_name={module_name}\tconfig={wrapper}\n")
print("target_spaces={}".format(target_spaces))
