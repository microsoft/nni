import random
import unittest
from collections import Counter

import nni.retiarii.nn.pytorch as nn
import torch
import torch.nn.functional as F
from nni.retiarii import Sampler, basic_unit
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.execution.python import _unpack_if_only_one
from nni.retiarii.nn.pytorch.mutator import process_inline_mutation, extract_mutation_from_pt_module
from nni.retiarii.serializer import model_wrapper
from nni.retiarii.utils import ContextStack


class EnumerateSampler(Sampler):
    def __init__(self):
        self.index = 0

    def choice(self, candidates, *args, **kwargs):
        choice = candidates[self.index % len(candidates)]
        self.index += 1
        return choice


class RandomSampler(Sampler):
    def __init__(self):
        self.counter = 0

    def choice(self, candidates, *args, **kwargs):
        self.counter += 1
        return random.choice(candidates)


@basic_unit
class MutableConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=1)

    def forward(self, x: torch.Tensor, index: int):
        if index == 0:
            return self.conv1(x)
        else:
            return self.conv2(x)


class GraphIR(unittest.TestCase):

    def _convert_to_ir(self, model):
        script_module = torch.jit.script(model)
        return convert_to_graph(script_module, model)

    def _get_converted_pytorch_model(self, model_ir):
        model_code = model_to_pytorch_script(model_ir)
        exec_vars = {}
        exec(model_code + '\n\nconverted_model = _model()', exec_vars)
        return exec_vars['converted_model']

    def _get_model_with_mutators(self, pytorch_model):
        model = self._convert_to_ir(pytorch_model)
        mutators = process_inline_mutation(model)
        return model, mutators

    def get_serializer(self):
        def dummy(cls):
            return cls

        return dummy

    def test_layer_choice(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.LayerChoice([
                    nn.Conv2d(3, 3, kernel_size=1),
                    nn.Conv2d(3, 5, kernel_size=1)
                ])

            def forward(self, x):
                return self.module(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 3, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 5, 3, 3]))

    def test_layer_choice_multiple(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.LayerChoice([nn.Conv2d(3, i, kernel_size=1) for i in range(1, 11)])

            def forward(self, x):
                return self.module(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        for i in range(1, 11):
            model_new = mutator.apply(model)
            self.assertEqual(self._get_converted_pytorch_model(model_new)(torch.randn(1, 3, 3, 3)).size(),
                             torch.Size([1, i, 3, 3]))

    def test_input_choice(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
                self.conv2 = nn.Conv2d(3, 5, kernel_size=1)
                self.input = nn.InputChoice(2)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                return self.input([x1, x2])

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 3, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 5, 3, 3]))

    def test_chosen_inputs(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self, reduction):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
                self.conv2 = nn.Conv2d(3, 3, kernel_size=1)
                self.input = nn.InputChoice(2, n_chosen=2, reduction=reduction)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                return self.input([x1, x2])

        for reduction in ['none', 'sum', 'mean', 'concat']:
            model, mutators = self._get_model_with_mutators(Net(reduction))
            self.assertEqual(len(mutators), 1)
            mutator = mutators[0].bind_sampler(EnumerateSampler())
            model = mutator.apply(model)
            result = self._get_converted_pytorch_model(model)(torch.randn(1, 3, 3, 3))
            if reduction == 'none':
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0].size(), torch.Size([1, 3, 3, 3]))
                self.assertEqual(result[1].size(), torch.Size([1, 3, 3, 3]))
            elif reduction == 'concat':
                self.assertEqual(result.size(), torch.Size([1, 6, 3, 3]))
            else:
                self.assertEqual(result.size(), torch.Size([1, 3, 3, 3]))

    def test_value_choice(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.index = nn.ValueChoice([0, 1])
                self.conv = MutableConv()

            def forward(self, x):
                return self.conv(x, self.index())

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 3, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 5, 3, 3]))

    def test_value_choice_as_parameter(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 5, kernel_size=nn.ValueChoice([3, 5]))

            def forward(self, x):
                return self.conv(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 1, 1]))

    def test_value_choice_as_parameter(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 5, kernel_size=nn.ValueChoice([3, 5]))

            def forward(self, x):
                return self.conv(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 1, 1]))

    def test_value_choice_as_parameter(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, nn.ValueChoice([6, 8]), kernel_size=nn.ValueChoice([3, 5]))

            def forward(self, x):
                return self.conv(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 2)
        mutators[0].bind_sampler(EnumerateSampler())
        mutators[1].bind_sampler(EnumerateSampler())
        input = torch.randn(1, 3, 5, 5)
        self.assertEqual(self._get_converted_pytorch_model(mutators[1].apply(mutators[0].apply(model)))(input).size(),
                         torch.Size([1, 6, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(mutators[1].apply(mutators[0].apply(model)))(input).size(),
                         torch.Size([1, 8, 1, 1]))

    def test_value_choice_as_parameter_shared(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, nn.ValueChoice([6, 8], label='shared'), 1)
                self.conv2 = nn.Conv2d(3, nn.ValueChoice([6, 8], label='shared'), 1)

            def forward(self, x):
                return self.conv1(x) + self.conv2(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 6, 5, 5]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 8, 5, 5]))

    def test_value_choice_in_functional(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout_rate = nn.ValueChoice([0., 1.])

            def forward(self, x):
                return F.dropout(x, self.dropout_rate())

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3))
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(), torch.Size([1, 3, 3, 3]))
        self.assertAlmostEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).abs().sum().item(), 0)

    def test_value_choice_in_layer_choice(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.LayerChoice([
                    nn.Linear(3, nn.ValueChoice([10, 20])),
                    nn.Linear(3, nn.ValueChoice([30, 40]))
                ])

            def forward(self, x):
                return self.linear(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 3)
        sz_counter = Counter()
        sampler = RandomSampler()
        for i in range(100):
            model_new = model
            for mutator in mutators:
                model_new = mutator.bind_sampler(sampler).apply(model_new)
            sz_counter[self._get_converted_pytorch_model(model_new)(torch.randn(1, 3)).size(1)] += 1
        self.assertEqual(len(sz_counter), 4)

    def test_shared(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self, shared=True):
                super().__init__()
                labels = ['x', 'x'] if shared else [None, None]
                self.module1 = nn.LayerChoice([
                    nn.Conv2d(3, 3, kernel_size=1),
                    nn.Conv2d(3, 5, kernel_size=1)
                ], label=labels[0])
                self.module2 = nn.LayerChoice([
                    nn.Conv2d(3, 3, kernel_size=1),
                    nn.Conv2d(3, 5, kernel_size=1)
                ], label=labels[1])

            def forward(self, x):
                return self.module1(x) + self.module2(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        sampler = RandomSampler()
        mutator = mutators[0].bind_sampler(sampler)
        self.assertEqual(self._get_converted_pytorch_model(mutator.apply(model))(torch.randn(1, 3, 3, 3)).size(0), 1)
        self.assertEqual(sampler.counter, 1)

        model, mutators = self._get_model_with_mutators(Net(shared=False))
        self.assertEqual(len(mutators), 2)
        sampler = RandomSampler()
        # repeat test. Expectation: sometimes succeeds, sometimes fails.
        failed_count = 0
        for i in range(30):
            model_new = model
            for mutator in mutators:
                model_new = mutator.bind_sampler(sampler).apply(model_new)
            self.assertEqual(sampler.counter, 2 * (i + 1))
            try:
                self._get_converted_pytorch_model(model_new)(torch.randn(1, 3, 3, 3))
            except RuntimeError:
                failed_count += 1
        self.assertGreater(failed_count, 0)
        self.assertLess(failed_count, 30)

    def test_valuechoice_access(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                vc = nn.ValueChoice([(6, 3), (8, 5)])
                self.conv = nn.Conv2d(3, vc[0], kernel_size=vc[1])

            def forward(self, x):
                return self.conv(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutators[0].bind_sampler(EnumerateSampler())
        input = torch.randn(1, 3, 5, 5)
        self.assertEqual(self._get_converted_pytorch_model(mutators[0].apply(model))(input).size(),
                         torch.Size([1, 6, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(mutators[0].apply(model))(input).size(),
                         torch.Size([1, 8, 1, 1]))

        @self.get_serializer()
        class Net2(nn.Module):
            def __init__(self):
                super().__init__()
                choices = [
                    {'b': [3], 'bp': [6]},
                    {'b': [6], 'bp': [12]}
                ]
                self.conv = nn.Conv2d(3, nn.ValueChoice(choices, label='a')['b'][0], 1)
                self.conv1 = nn.Conv2d(nn.ValueChoice(choices, label='a')['bp'][0], 3, 1)

            def forward(self, x):
                x = self.conv(x)
                return self.conv1(torch.cat((x, x), 1))

        model, mutators = self._get_model_with_mutators(Net2())
        self.assertEqual(len(mutators), 1)
        mutators[0].bind_sampler(EnumerateSampler())
        input = torch.randn(1, 3, 5, 5)
        self._get_converted_pytorch_model(mutators[0].apply(model))(input)

    def test_valuechoice_access_functional(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout_rate = nn.ValueChoice([[0., ], [1., ]])

            def forward(self, x):
                return F.dropout(x, self.dropout_rate()[0])

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3))
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(), torch.Size([1, 3, 3, 3]))
        self.assertAlmostEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).abs().sum().item(), 0)

    def test_valuechoice_access_functional_expression(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout_rate = nn.ValueChoice([[1.05, ], [1.1, ]])

            def forward(self, x):
                # if expression failed, the exception would be:
                # ValueError: dropout probability has to be between 0 and 1, but got 1.05
                return F.dropout(x, self.dropout_rate()[0] - .1)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3))
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(), torch.Size([1, 3, 3, 3]))
        self.assertAlmostEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).abs().sum().item(), 0)

    def test_repeat(self):
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1

        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Repeat(AddOne(), (3, 5))

            def forward(self, x):
                return self.block(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        model3 = mutator.apply(model)
        self.assertTrue((self._get_converted_pytorch_model(model1)(torch.zeros(1, 16)) == 3).all())
        self.assertTrue((self._get_converted_pytorch_model(model2)(torch.zeros(1, 16)) == 4).all())
        self.assertTrue((self._get_converted_pytorch_model(model3)(torch.zeros(1, 16)) == 5).all())

    def test_cell(self):
        @self.get_serializer()
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                                    num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

            def forward(self, x, y):
                return self.cell([x, y])

        raw_model, mutators = self._get_model_with_mutators(Net())
        for _ in range(10):
            sampler = EnumerateSampler()
            model = raw_model
            for mutator in mutators:
                model = mutator.bind_sampler(sampler).apply(model)
            self.assertTrue(self._get_converted_pytorch_model(model)(
                torch.randn(1, 16), torch.randn(1, 16)).size() == torch.Size([1, 64]))

        @self.get_serializer()
        class Net2(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)], num_nodes=4)

            def forward(self, x):
                return self.cell([x])

        raw_model, mutators = self._get_model_with_mutators(Net2())
        for _ in range(10):
            sampler = EnumerateSampler()
            model = raw_model
            for mutator in mutators:
                model = mutator.bind_sampler(sampler).apply(model)
            self.assertTrue(self._get_converted_pytorch_model(model)(torch.randn(1, 16)).size() == torch.Size([1, 64]))


class Python(GraphIR):
    def _get_converted_pytorch_model(self, model_ir):
        mutation = {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model_ir.history}
        with ContextStack('fixed', mutation):
            model = model_ir.python_class(**model_ir.python_init_params)
            return model

    def _get_model_with_mutators(self, pytorch_model):
        return extract_mutation_from_pt_module(pytorch_model)

    def get_serializer(self):
        return model_wrapper

    @unittest.skip
    def test_value_choice(self): ...

    @unittest.skip
    def test_value_choice_in_functional(self): ...

    @unittest.skip
    def test_valuechoice_access_functional(self): ...

    @unittest.skip
    def test_valuechoice_access_functional_expression(self): ...
