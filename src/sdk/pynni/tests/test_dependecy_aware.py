# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest
from unittest import TestCase, main
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from nni.compression.torch import L1FilterPruner, L2FilterPruner, FPGMPruner, \
                                  TaylorFOWeightFilterPruner, ActivationAPoZRankFilterPruner, \
                                  ActivationMeanRankFilterPruner
from nni.compression.torch import ModelSpeedup

unittest.TestLoader.sortTestMethodsUsing = None

MODEL_FILE, MASK_FILE = './model.pth', './mask.pth'


class DependencyawareTest(TestCase):
    @unittest.skipIf(torch.__version__ < "1.3.0", "not supported")
    def test_dependency_aware_pruning(self):
        model_zoo = ['resnet18']
        pruners = [ActivationAPoZRankFilterPruner]
        sparsity = 0.7
        cfg_list = [{'op_types': ['Conv2d'], 'sparsity':sparsity}]
        dummy_input = torch.ones(1, 3, 224, 224)
        for model_name in model_zoo:
            for pruner in pruners:
                ori_filters = {}
                Model = getattr(models, model_name)
                net = Model(pretrained=True, progress=False)
                # record the number of the filter of each conv layer
                for name, module in net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        ori_filters[name] = module.out_channels
                tmp_pruner = pruner(
                    net, cfg_list, dependency_aware=True, dummy_input=dummy_input)
                # for the pruners that based on the activations, we need feed
                # enough data before we call the compress function.
                net(dummy_input)
                tmp_pruner.compress()
                tmp_pruner.export_model(MODEL_FILE, MASK_FILE)
                # if we want to use the same model, we should unwrap the pruner before the speedup
                tmp_pruner._unwrap_model()
                ms = ModelSpeedup(net, dummy_input, MASK_FILE)
                ms.speedup_model()
                for name, module in net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        filter_diff = abs(
                            int(ori_filters[name] * (1-sparsity)) - module.out_channels)
                        # because we are using the dependency-aware mode, so the number of the
                        # filters after speedup should be ori_filters[name] * ( 1 - sparsity )
                        assert filter_diff <= 1


if __name__ == '__main__':
    main()
