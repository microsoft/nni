# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest import TestCase, main

import torch
from nni.nas.pytorch.enas import EnasMutator
from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.random import RandomMutator

from .models.pytorch import NaiveSearchSpace, NestedSpace, SpaceWithMutableScope


class NasTestCase(TestCase):
    def setUp(self):
        self.input_size = [3, 32, 32]

    def iterative_sample_and_forward(self, model, mutator, input_size, n_iters=20, test_backward=True,
                                     cuda_only=False, parallel_mode=None):
        # support pytorch only
        input_size = [2] + input_size  # 2 samples to enable batch norm
        for _ in range(n_iters):
            mutator.reset()
            x = torch.randn(input_size)
            y = torch.sum(model(x))
            if test_backward:
                y.backward()

    def test_random_mutator(self):
        for model_cls in [NaiveSearchSpace, SpaceWithMutableScope]:
            model = model_cls(self)
            mutator = RandomMutator(model)
            self.iterative_sample_and_forward(model, mutator, self.input_size)

    def test_enas_mutator(self):
        for model_cls in [NaiveSearchSpace, SpaceWithMutableScope]:
            model = model_cls(self)
            mutator = EnasMutator(model)
            self.iterative_sample_and_forward(model, mutator, self.input_size)

    def test_darts_mutator(self):
        for model_cls in [NaiveSearchSpace, SpaceWithMutableScope]:
            model = model_cls(self)
            mutator = DartsMutator(model)
            self.iterative_sample_and_forward(model, mutator, self.input_size)

    def test_nested_space(self):
        model = NestedSpace(self)
        with self.assertRaises(RuntimeError):
            RandomMutator(model)


if __name__ == '__main__':
    main()
