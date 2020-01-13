# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import importlib
import os
import sys
from unittest import TestCase, main

import torch
from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.enas import EnasMutator
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.random import RandomMutator
from nni.nas.pytorch.utils import reset_global_mutable_counting


class NasTestCase(TestCase):

    def setUp(self):
        self.default_input_size = [3, 32, 32]
        self.model_path = os.path.join(os.path.dirname(__file__), "models")
        sys.path.append(self.model_path)
        self.model_module = importlib.import_module("pytorch_models")
        self.default_cls = [self.model_module.NaiveSearchSpace, self.model_module.SpaceWithMutableScope]

    def tearDown(self):
        sys.path.remove(self.model_path)

    def iterative_sample_and_forward(self, model, mutator=None, input_size=None, n_iters=20, test_backward=True):
        if input_size is None:
            input_size = self.default_input_size
        # support pytorch only
        input_size = [2] + input_size  # 2 samples to enable batch norm
        for _ in range(n_iters):
            for param in model.parameters():
                param.grad = None
            if mutator is not None:
                mutator.reset()
            x = torch.randn(input_size)
            y = torch.sum(model(x))
            if test_backward:
                y.backward()

    def default_mutator_test_pipeline(self, mutator_cls):
        for model_cls in self.default_cls:
            reset_global_mutable_counting()
            model = model_cls(self)
            mutator = mutator_cls(model)
            self.iterative_sample_and_forward(model, mutator)
            reset_global_mutable_counting()
            model_fixed = model_cls(self)
            with torch.no_grad():
                arc = mutator.export()
            apply_fixed_architecture(model_fixed, arc)
            self.iterative_sample_and_forward(model_fixed, n_iters=1)

    def test_random_mutator(self):
        self.default_mutator_test_pipeline(RandomMutator)

    def test_enas_mutator(self):
        self.default_mutator_test_pipeline(EnasMutator)

    def test_darts_mutator(self):
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


if __name__ == '__main__':
    main()
