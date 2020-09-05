import logging
import random
import unittest

import torch

from sdk import strategy, utils
from sdk.graph import Graph

from examples.allstars.searchspace import (chamnet, fbnet, nasbench101, nasbench201, nasnet,
                                           nasrnn, onceforall, proxylessnas, singlepathnas)


class CodeGenTest(unittest.TestCase):

    def _codegen_utils(self, graph, mutators, dependencies, dummy_input):
        class RandomSampler(strategy.Sampler):
            def choice(self, candidates):
                return random.choice(candidates)

        base_graph = Graph.load(graph)
        sampler = RandomSampler()
        graphs = []
        for i in range(5):
            new_graph = base_graph.duplicate()
            for mutator in mutators:
                new_graph = mutator.apply(new_graph, sampler)
            new_graph.generate_code('pytorch', output_file=f'generated/graph_{i}.py')

            # hack: add import part
            with open(f'generated/graph_{i}.py') as f:
                code = f.read()
            code = ''.join([f'from {d} import *\n' for d in dependencies]) + code
            with open(f'generated/graph_{i}.py', 'w') as f:
                f.write(code)

            model_cls = utils.import_(f'generated.graph_{i}.Graph')
            model = model_cls()
            model(dummy_input)

    def test_fbnet(self):
        model, mutators = fbnet()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.fbnet'], torch.randn(1, 3, 224, 224))

    def test_nasnet(self):
        # nasnet, amoebanet, pnas
        model, mutators = nasnet()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.nasnet'], torch.randn(1, 3, 224, 224))

    def test_nasrnn(self):
        model, mutators = nasrnn()
        self._codegen_utils(model, mutators, [], (torch.randint(5, (30, 1)), torch.randint(1, 128)))

    def test_nasbench101(self):
        model, mutators = nasbench101()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.nasbench101'], torch.randn(1, 3, 32, 32))

    def test_nasbench201(self):
        model, mutators = nasbench201()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.nasbench201'], torch.randn(1, 3, 32, 32))

    def test_proxylessnas(self):
        model, mutators = proxylessnas()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.proxylessnas'], torch.randn(1, 3, 224, 224))

    def test_chamnet(self):
        model, mutators = chamnet()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.proxylessnas'], torch.randn(1, 3, 224, 224))

    def test_onceforall(self):
        model, mutators = onceforall()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.proxylessnas'], torch.randn(1, 3, 224, 224))

    def test_singlepathnas(self):
        model, mutators = singlepathnas()
        self._codegen_utils(model, mutators, ['examples.allstars.searchspace.proxylessnas'], torch.randn(1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
