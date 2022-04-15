import logging
import os
import gc
import copy
import psutil
import sys
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, vgg11
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenet import mobilenet_v2
import unittest
from unittest import TestCase, main

from nni.compression.pytorch import ModelSpeedup, apply_compression_results
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.algorithms.compression.pytorch.pruning.weight_masker import WeightMasker
from nni.algorithms.compression.pytorch.pruning.dependency_aware_pruner import DependencyAwarePruner
from mobilenet import *
# need propagation branch of nni
ori_model = MobileNet(120).cuda()
dummy_input = torch.rand(8, 3 , 224, 224).cuda()
names = []
for name, module in ori_model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        names.append(name)
ref = open('finegrained_propagate.csv', 'w')
ref.write(f"Sparsity,{','.join(names)}\n")
for sparsity_ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
    print(sparsity_ratio)
    cfg_list = [{'sparsity': sparsity_ratio, 'op_types':['Conv2d', 'Linear']}]
    model = copy.deepcopy(ori_model)
    pruner = LevelPruner(model, cfg_list)
    pruner.compress()
    pruner.export_model('./weight.pth', './mask.pth')
    pruner._unwrap_model()
    ms = ModelSpeedup(model, dummy_input, './mask.pth')
    new_mask = ms.propagate_mask()
    ref.write(str(sparsity_ratio))
    # sparsities = {}
    for _n in names:
        new_sparsity = 1 - torch.sum(new_mask[_n]['weight'])/new_mask[_n]['weight'].numel()
        new_sparsity = max(sparsity_ratio, new_sparsity)
        ref.write(f',%.4f' % new_sparsity)
    ref.write('\n')
ref.close()
# print(model)