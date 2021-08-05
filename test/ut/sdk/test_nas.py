# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import importlib
import os
import sys
from collections import OrderedDict
from unittest import TestCase, main

import torch
import torch.nn as nn
from nni.algorithms.nas.pytorch.classic_nas import get_and_apply_next_architecture
from nni.algorithms.nas.pytorch.darts import DartsMutator
from nni.algorithms.nas.pytorch.enas import EnasMutator
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.mutables import LayerChoice
from nni.algorithms.nas.pytorch.random import RandomMutator
from nni.nas.pytorch.utils import _reset_global_mutable_counting


class NasTestCase(TestCase):

    def setUp(self):
        self.default_input_size = [3, 32, 32]
        self.model_path = os.path.join(os.path.dirname(__file__), "models")
        sys.path.append(self.model_path)
        self.model_module = importlib.import_module("pytorch_models")
        self.default_cls = [self.model_module.NaiveSearchSpace, self.model_module.SpaceWithMutableScope]
        self.cuda_test = [0]
        if torch.cuda.is_available():
            self.cuda_test.append(1)
        if torch.cuda.device_count() > 1:
            self.cuda_test.append(torch.cuda.device_count())

    def tearDown(self):
        sys.path.remove(self.model_path)

    def iterative_sample_and_forward(self, model, mutator=None, input_size=None, n_iters=20, test_backward=True,
                                     use_cuda=False):
        if input_size is None:
            input_size = self.default_input_size
        # support pytorch only
        input_size = [8 if use_cuda else 2] + input_size  # at least 2 samples to enable batch norm
        for _ in range(n_iters):
            for param in model.parameters():
                param.grad = None
            if mutator is not None:
                mutator.reset()
            x = torch.randn(input_size)
            if use_cuda:
                x = x.cuda()
            y = torch.sum(model(x))
            if test_backward:
                y.backward()

    def default_mutator_test_pipeline(self, mutator_cls):
        for model_cls in self.default_cls:
            for cuda_test in self.cuda_test:
                _reset_global_mutable_counting()
                model = model_cls(self)
                mutator = mutator_cls(model)
                if cuda_test:
                    model.cuda()
                    mutator.cuda()
                    if cuda_test > 1:
                        model = nn.DataParallel(model)
                self.iterative_sample_and_forward(model, mutator, use_cuda=cuda_test)
                _reset_global_mutable_counting()
                model_fixed = model_cls(self)
                if cuda_test:
                    model_fixed.cuda()
                    if cuda_test > 1:
                        model_fixed = nn.DataParallel(model_fixed)
                with torch.no_grad():
                    arc = mutator.export()
                apply_fixed_architecture(model_fixed, arc)
                self.iterative_sample_and_forward(model_fixed, n_iters=1, use_cuda=cuda_test)

    def test_random_mutator(self):
        self.default_mutator_test_pipeline(RandomMutator)

    def test_enas_mutator(self):
        self.default_mutator_test_pipeline(EnasMutator)

    def test_darts_mutator(self):
        # DARTS doesn't support DataParallel. To be fixed.
        self.cuda_test = [t for t in self.cuda_test if t <= 1]
        self.default_mutator_test_pipeline(DartsMutator)

    def test_apply_twice(self):
        model = self.model_module.NaiveSearchSpace(self)
        with self.assertRaises(RuntimeError):
            for _ in range(2):
                RandomMutator(model)

    def test_nested_space(self):
        model = self.model_module.NestedSpace(self)
        with self.assertRaises(RuntimeError):
            RandomMutator(model)

    def test_classic_nas(self):
        for model_cls in self.default_cls:
            model = model_cls(self)
            get_and_apply_next_architecture(model)
            self.iterative_sample_and_forward(model)

    def test_proxylessnas(self):
        model = self.model_module.LayerChoiceOnlySearchSpace(self)
        get_and_apply_next_architecture(model)
        self.iterative_sample_and_forward(model)

    def test_layer_choice(self):
        for i in range(2):
            for j in range(2):
                if j == 0:
                    # test number
                    layer_choice = LayerChoice([nn.Conv2d(3, 3, 3), nn.Conv2d(3, 5, 3), nn.Conv2d(3, 6, 3)])
                else:
                    # test ordered dict
                    layer_choice = LayerChoice(OrderedDict([
                        ("conv1", nn.Conv2d(3, 3, 3)),
                        ("conv2", nn.Conv2d(3, 5, 3)),
                        ("conv3", nn.Conv2d(3, 6, 3))
                    ]))
                if i == 0:
                    # test modify
                    self.assertEqual(len(layer_choice.choices), 3)
                    layer_choice[1] = nn.Conv2d(3, 4, 3)
                    self.assertEqual(layer_choice[1].out_channels, 4)
                    self.assertEqual(len(layer_choice[0:2]), 2)
                    if j > 0:
                        layer_choice["conv3"] = nn.Conv2d(3, 7, 3)
                        self.assertEqual(layer_choice[-1].out_channels, 7)
                if i == 1:
                    # test delete
                    del layer_choice[1]
                    self.assertEqual(len(layer_choice), 2)
                    self.assertEqual(len(list(layer_choice)), 2)
                    self.assertEqual(layer_choice.names, ["conv1", "conv3"] if j > 0 else ["0", "2"])
                    if j > 0:
                        del layer_choice["conv1"]
                        self.assertEqual(len(layer_choice), 1)


if __name__ == '__main__':
    main()
