# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet18
from unittest import TestCase, main

from nni.compression.torch import L1FilterPruner, apply_compression_results, ModelSpeedup

torch.manual_seed(0)

class BackboneModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, 1)
    def forward(self, x):
        return self.conv1(x)

class BackboneModel2(torch.nn.Module):
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

class BigModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone1 = BackboneModel1()
        self.backbone2 = BackboneModel2()
        self.fc3 =  nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2)
        )
    def forward(self, x):
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.fc3(x)
        return x

dummy_input = torch.randn(2, 1, 28, 28)
SPARSITY = 0.5
MODEL_FILE, MASK_FILE = './11_model.pth', './l1_mask.pth'

def prune_model_l1(model):
    config_list = [{
        'sparsity': SPARSITY,
        'op_types': ['Conv2d']
    }]
    pruner = L1FilterPruner(model, config_list)
    pruner.compress()
    pruner.export_model(model_path=MODEL_FILE, mask_path=MASK_FILE)

class SpeedupTestCase(TestCase):
    def test_speedup_vgg16(self):
        prune_model_l1(vgg16())
        model = vgg16()
        model.train()
        ms = ModelSpeedup(model, torch.randn(2, 3, 32, 32), MASK_FILE)
        ms.speedup_model()

        orig_model = vgg16()
        assert model.training
        assert model.features[2].out_channels == int(orig_model.features[2].out_channels * SPARSITY)
        assert model.classifier[0].in_features == int(orig_model.classifier[0].in_features * SPARSITY)

    #def test_speedup_resnet(self):
        #TODO support resnet
        #model = resnet18()

    def test_speedup_bigmodel(self):
        prune_model_l1(BigModel())
        model = BigModel()
        apply_compression_results(model, MASK_FILE, 'cpu')
        model.eval()
        mask_out = model(dummy_input)

        model.train()
        ms = ModelSpeedup(model, dummy_input, MASK_FILE)
        ms.speedup_model()
        assert model.training

        model.eval()
        speedup_out = model(dummy_input)
        if not torch.allclose(mask_out, speedup_out, atol=1e-07):
            print('input:', dummy_input.size(), torch.abs(dummy_input).sum((2,3)))
            print('mask_out:', mask_out)
            print('speedup_out:', speedup_out)
            raise RuntimeError('model speedup inference result is incorrect!')

        orig_model = BigModel()

        assert model.backbone2.conv1.out_channels == int(orig_model.backbone2.conv1.out_channels * SPARSITY)
        assert model.backbone2.conv2.in_channels == int(orig_model.backbone2.conv2.in_channels * SPARSITY)
        assert model.backbone2.conv2.out_channels == int(orig_model.backbone2.conv2.out_channels * SPARSITY)
        assert model.backbone2.fc1.in_features == int(orig_model.backbone2.fc1.in_features * SPARSITY)

    def tearDown(self):
        os.remove(MODEL_FILE)
        os.remove(MASK_FILE)

if __name__ == '__main__':
    main()
