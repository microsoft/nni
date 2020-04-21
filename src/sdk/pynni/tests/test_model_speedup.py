# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from unittest import TestCase, main

from nni.graph_utils import TorchGraph
from nni.compression.torch import L1FilterPruner
from nni.compression.speedup.torch import ModelSpeedup

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, 1)
    def forward(self, x):
        return self.conv1(x)

class BigModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = NaiveModel()
        self.backbone2 = Model2()
        self.fc3 = nn.Linear(10, 2) 
    def forward(self, x):
        x = self.backbone2(x)
        x = self.backbone(x)
        x = self.fc3(x)
        return x


class TorchGraphTestCase(TestCase):
    def test_build_graph(self):
        big_model = BigModel()
        g = TorchGraph(big_model, torch.randn(2, 1, 28, 28))
        print(g.leaf_modules)
        print([x for x in g.name_to_gnode])
        leaf_modules = set([
            'backbone.bn1', 'backbone.bn2', 'backbone.conv1', 'backbone.conv2',
            'backbone.fc1', 'backbone.fc2', 'backbone2.conv1', 'fc3'])
        assert set(g.leaf_modules) == leaf_modules
        assert g._find_successors('backbone.conv1') == ['backbone.bn1']
        assert g._find_successors('backbone.conv2') == ['backbone.bn2']
        assert g._find_predecessors('backbone.bn1') == ['backbone.conv1']
        assert g._find_predecessors('backbone.bn2') == ['backbone.conv2']

    def test_speedup(self):
        config_list = [{
            'sparsity': 0.5,
            'op_types': ['Conv2d']
        }]
        model = BigModel()
        pruner = L1FilterPruner(model, config_list)
        pruner.compress()
        pruner.export_model(model_path='./11_model.pth', mask_path='./l1_mask.pth')

        try:
            model = BigModel()
            ms = ModelSpeedup(model, torch.randn(2, 1, 28, 28), './l1_mask.pth')
            ms.speedup_model()
            assert model.backbone.conv2.in_channels == 10
        finally:
            os.remove('./11_model.pth')
            os.remove('./l1_mask.pth')

if __name__ == '__main__':
    main()
