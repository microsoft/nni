# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import random
import unittest
from unittest import TestCase, main
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, L2FilterPruner, FPGMPruner, \
    TaylorFOWeightFilterPruner, ActivationAPoZRankFilterPruner, \
    ActivationMeanRankFilterPruner
from nni.compression.pytorch import ModelSpeedup

unittest.TestLoader.sortTestMethodsUsing = None

MODEL_FILE, MASK_FILE = './model.pth', './mask.pth'

def generate_random_sparsity(model):
    """
    generate a random sparsity for all conv layers in the
    model.
    """
    cfg_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            sparsity = np.random.uniform(0.5, 0.99)
            cfg_list.append({'op_types': ['Conv2d'], 'op_names': [name],
                             'sparsity': sparsity})
    return cfg_list

def generate_random_sparsity_v2(model):
    """
    only generate a random sparsity for some conv layers in
    in the model.
    """
    cfg_list = []
    for name, module in model.named_modules():
        # randomly pick 50% layers
        if isinstance(module, nn.Conv2d) and random.uniform(0, 1) > 0.5:
            sparsity = np.random.uniform(0.5, 0.99)
            cfg_list.append({'op_types': ['Conv2d'], 'op_names': [name],
                             'sparsity': sparsity})
    return cfg_list


class DependencyawareTest(TestCase):
    @unittest.skipIf(torch.__version__ < "1.3.0", "not supported")
    def test_dependency_aware_pruning(self):
        model_zoo = ['resnet18']
        pruners = [L1FilterPruner, L2FilterPruner, FPGMPruner, TaylorFOWeightFilterPruner]
        sparsity = 0.7
        cfg_list = [{'op_types': ['Conv2d'], 'sparsity':sparsity}]
        dummy_input = torch.ones(1, 3, 224, 224)
        for model_name in model_zoo:
            for pruner in pruners:
                print('Testing on ', pruner)
                ori_filters = {}
                Model = getattr(models, model_name)
                net = Model(pretrained=True, progress=False)
                # record the number of the filter of each conv layer
                for name, module in net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        ori_filters[name] = module.out_channels

                # for the pruners that based on the activations, we need feed
                # enough data before we call the compress function.
                optimizer = torch.optim.SGD(net.parameters(), lr=0.0001,
                                 momentum=0.9,
                                 weight_decay=4e-5)
                criterion = torch.nn.CrossEntropyLoss()
                tmp_pruner = pruner(
                    net, cfg_list, optimizer, dependency_aware=True, dummy_input=dummy_input)
                # train one single batch so that the the pruner can collect the
                # statistic
                optimizer.zero_grad()
                out = net(dummy_input)
                batchsize = dummy_input.size(0)
                loss = criterion(out, torch.zeros(batchsize, dtype=torch.int64))
                loss.backward()
                optimizer.step()

                tmp_pruner.compress()
                tmp_pruner.export_model(MODEL_FILE, MASK_FILE)
                # if we want to use the same model, we should unwrap the pruner before the speedup
                tmp_pruner._unwrap_model()
                ms = ModelSpeedup(net, dummy_input, MASK_FILE)
                ms.speedup_model()
                for name, module in net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        expected = int(ori_filters[name] * (1-sparsity))
                        filter_diff = abs(expected - module.out_channels)
                        errmsg = '%s Ori: %d, Expected: %d, Real: %d' % (
                            name, ori_filters[name], expected, module.out_channels)

                        # because we are using the dependency-aware mode, so the number of the
                        # filters after speedup should be ori_filters[name] * ( 1 - sparsity )
                        print(errmsg)
                        assert filter_diff <= 1, errmsg

    @unittest.skipIf(torch.__version__ < "1.3.0", "not supported")
    def test_dependency_aware_random_config(self):
        model_zoo = ['resnet18']
        pruners = [L1FilterPruner, L2FilterPruner, FPGMPruner, TaylorFOWeightFilterPruner,
                   ActivationMeanRankFilterPruner, ActivationAPoZRankFilterPruner]
        dummy_input = torch.ones(1, 3, 224, 224)
        for model_name in model_zoo:
            for pruner in pruners:
                Model = getattr(models, model_name)
                cfg_generator = [generate_random_sparsity, generate_random_sparsity_v2]
                for _generator in cfg_generator:
                    net = Model(pretrained=True, progress=False)
                    cfg_list = _generator(net)

                    print('\n\nModel:', model_name)
                    print('Pruner', pruner)
                    print('Config_list:', cfg_list)
                    # for the pruners that based on the activations, we need feed
                    # enough data before we call the compress function.
                    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001,
                                    momentum=0.9,
                                    weight_decay=4e-5)
                    criterion = torch.nn.CrossEntropyLoss()
                    tmp_pruner = pruner(
                        net, cfg_list, optimizer, dependency_aware=True, dummy_input=dummy_input)
                    # train one single batch so that the the pruner can collect the
                    # statistic
                    optimizer.zero_grad()
                    out = net(dummy_input)
                    batchsize = dummy_input.size(0)
                    loss = criterion(out, torch.zeros(batchsize, dtype=torch.int64))
                    loss.backward()
                    optimizer.step()

                    tmp_pruner.compress()
                    tmp_pruner.export_model(MODEL_FILE, MASK_FILE)
                    # if we want to use the same model, we should unwrap the pruner before the speedup
                    tmp_pruner._unwrap_model()
                    ms = ModelSpeedup(net, dummy_input, MASK_FILE)
                    ms.speedup_model()


if __name__ == '__main__':
    main()
