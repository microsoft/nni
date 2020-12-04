# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet18
import unittest
from unittest import TestCase, main

from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, apply_compression_results
from nni.algorithms.compression.pytorch.pruning.weight_masker import WeightMasker
from nni.algorithms.compression.pytorch.pruning.one_shot import _StructuredFilterPruner

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
# the relative distance
RELATIVE_THRESHOLD = 0.01
# Because of the precision of floating-point numbers, some errors
# between the original output tensors(without speedup) and the output
# tensors of the speedup model are normal. When the output tensor itself
# is small, such errors may exceed the relative threshold, so we also add
# an absolute threshold to determine whether the final result is correct.
# The error should meet the RELATIVE_THREHOLD or the ABSOLUTE_THRESHOLD.
ABSOLUTE_THRESHOLD = 0.0001
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

def generate_random_sparsity(model):
    cfg_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            sparsity = np.random.uniform(0.5, 0.99)
            cfg_list.append({'op_types': ['Conv2d'], 'op_names': [name],
                             'sparsity': sparsity})
    return cfg_list

def zero_bn_bias(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) \
            or isinstance(module, nn.BatchNorm3d) \
            or isinstance(module, nn.BatchNorm1d):
                shape = module.bias.data.size()
                device = module.bias.device
                module.bias.data = torch.zeros(shape).to(device)
                shape = module.running_mean.data.size()
                module.running_mean = torch.zeros(shape).to(device)

class L1ChannelMasker(WeightMasker):
    def __init__(self, model, pruner):
        self.model = model
        self.pruner = pruner

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        msg = 'module type {} is not supported!'.format(wrapper.type)
        #assert wrapper.type == 'Conv2d', msg
        weight = wrapper.module.weight.data
        bias = None
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            bias = wrapper.module.bias.data

        if wrapper.weight_mask is None:
            mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        else:
            mask_weight = wrapper.weight_mask.clone()
        if bias is not None:
            if wrapper.bias_mask is None:
                mask_bias = torch.ones(bias.size()).type_as(bias).detach()
            else:
                mask_bias = wrapper.bias_mask.clone()
        else:
            mask_bias = None
        base_mask = {'weight_mask': mask_weight, 'bias_mask': mask_bias}

        num_total = weight.size(1)
        num_prune = int(num_total * sparsity)

        if num_total < 2 or num_prune < 1:
            return base_mask
        w_abs = weight.abs()
        if wrapper.type == 'Conv2d':
            w_abs_structured = w_abs.sum((0, 2, 3))
            threshold = torch.topk(w_abs_structured, num_prune, largest=False)[0].max()
            mask_weight = torch.gt(w_abs_structured, threshold)[None, :, None, None].expand_as(weight).type_as(weight)
            return {'weight_mask': mask_weight.detach()}
        else:
            # Linear
            assert wrapper.type == 'Linear'
            w_abs_structured = w_abs.sum((0))
            threshold = torch.topk(w_abs_structured, num_prune, largest=False)[0].max()
            mask_weight = torch.gt(w_abs_structured, threshold)[None, :].expand_as(weight).type_as(weight)
            return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}

class L1ChannelPruner(_StructuredFilterPruner):
    def __init__(self, model, config_list, optimizer=None, dependency_aware=False, dummy_input=None):
        super().__init__(model, config_list, pruning_algorithm='l1', optimizer=optimizer,
                         dependency_aware=dependency_aware, dummy_input=dummy_input)
    def validate_config(self, model, config_list):
        pass


def channel_prune(model):
    config_list = [{
        'sparsity': SPARSITY,
        'op_types': ['Conv2d', 'Linear']
    }, {
        'op_names': ['conv1'],
        'exclude': True
    }]

    pruner = L1ChannelPruner(model, config_list)
    masker = L1ChannelMasker(model, pruner)
    pruner.masker = masker
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

    # FIXME: This test case might fail randomly, no idea why
    # Example: https://msrasrg.visualstudio.com/NNIOpenSource/_build/results?buildId=16282

    def test_speedup_integration(self):
        for model_name in ['resnet18', 'squeezenet1_1', 'mobilenet_v2', 'densenet121', 'densenet169', 'inception_v3', 'resnet50']:
            kwargs = {
                'pretrained': True
            }
            if model_name == 'resnet50':
                # testing multiple groups
                kwargs = {
                    'pretrained': False,
                    'groups': 4
                }

            Model = getattr(models, model_name)
            net = Model(**kwargs).to(device)
            speedup_model = Model(**kwargs).to(device)
            net.eval() # this line is necessary
            speedup_model.eval()
            # random generate the prune config for the pruner
            cfgs = generate_random_sparsity(net)
            pruner = L1FilterPruner(net, cfgs)
            pruner.compress()
            pruner.export_model(MODEL_FILE, MASK_FILE)
            pruner._unwrap_model()
            state_dict = torch.load(MODEL_FILE)
            speedup_model.load_state_dict(state_dict)
            zero_bn_bias(net)
            zero_bn_bias(speedup_model)

            data = torch.ones(BATCH_SIZE, 3, 128, 128).to(device)
            ms = ModelSpeedup(speedup_model, data, MASK_FILE)
            ms.speedup_model()

            speedup_model.eval()

            ori_out = net(data)
            speeded_out = speedup_model(data)
            ori_sum = torch.sum(ori_out).item()
            speeded_sum = torch.sum(speeded_out).item()
            print('Sum of the output of %s (before speedup):'%model_name, ori_sum)
            print('Sum of the output of %s (after speedup):'%model_name, speeded_sum)
            assert (abs(ori_sum - speeded_sum) / abs(ori_sum) < RELATIVE_THRESHOLD) or \
                   (abs(ori_sum - speeded_sum) < ABSOLUTE_THRESHOLD)

    def test_channel_prune(self):
        orig_net = resnet18(num_classes=10).to(device)
        channel_prune(orig_net)
        state_dict = torch.load(MODEL_FILE)

        orig_net = resnet18(num_classes=10).to(device)
        orig_net.load_state_dict(state_dict)
        apply_compression_results(orig_net, MASK_FILE)
        orig_net.eval()

        net = resnet18(num_classes=10).to(device)

        net.load_state_dict(state_dict)
        net.eval()

        data = torch.randn(BATCH_SIZE, 3, 128, 128).to(device)
        ms = ModelSpeedup(net, data, MASK_FILE)
        ms.speedup_model()
        ms.bound_model(data)

        net.eval()

        ori_sum = orig_net(data).abs().sum().item()
        speeded_sum = net(data).abs().sum().item()

        print(ori_sum, speeded_sum)
        assert (abs(ori_sum - speeded_sum) / abs(ori_sum) < RELATIVE_THRESHOLD) or \
            (abs(ori_sum - speeded_sum) < ABSOLUTE_THRESHOLD)

    def tearDown(self):
        os.remove(MODEL_FILE)
        os.remove(MASK_FILE)

if __name__ == '__main__':
    main()
