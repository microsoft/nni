import random
import unittest
from collections import Counter

import nni.retiarii.nn.pytorch as nn
import torch
import torch.nn.functional as F
from nni.retiarii import InvalidMutation, Sampler, basic_unit
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.execution.utils import _unpack_if_only_one
from nni.retiarii.graph import Model
from nni.retiarii.nn.pytorch.api import ValueChoice
from nni.retiarii.nn.pytorch.mutator import process_evaluator_mutations, process_inline_mutation, extract_mutation_from_pt_module
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

    def test_layer_choice(self):
        @model_wrapper
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
        @model_wrapper
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

    def test_nested_layer_choice(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.LayerChoice([
                    nn.LayerChoice([nn.Conv2d(3, 3, kernel_size=1),
                                    nn.Conv2d(3, 4, kernel_size=1),
                                    nn.Conv2d(3, 5, kernel_size=1)]),
                    nn.Conv2d(3, 1, kernel_size=1)
                ])

            def forward(self, x):
                return self.module(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 2)
        mutators[0].bind_sampler(EnumerateSampler())
        mutators[1].bind_sampler(EnumerateSampler())
        input = torch.randn(1, 3, 5, 5)
        self.assertEqual(self._get_converted_pytorch_model(mutators[1].apply(mutators[0].apply(model)))(input).size(),
                         torch.Size([1, 3, 5, 5]))
        self.assertEqual(self._get_converted_pytorch_model(mutators[1].apply(mutators[0].apply(model)))(input).size(),
                         torch.Size([1, 1, 5, 5]))
        self.assertEqual(self._get_converted_pytorch_model(mutators[1].apply(mutators[0].apply(model)))(input).size(),
                         torch.Size([1, 5, 5, 5]))

    def test_input_choice(self):
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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

        @model_wrapper
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
        @model_wrapper
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
        @model_wrapper
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

        @model_wrapper
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

    def test_repeat_complex(self):
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1

        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Repeat(nn.LayerChoice([AddOne(), nn.Identity()], label='lc'), (3, 5), label='rep')

            def forward(self, x):
                return self.block(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 2)
        self.assertEqual(set([mutator.label for mutator in mutators]), {'lc', 'rep'})

        sampler = RandomSampler()
        for _ in range(10):
            new_model = model
            for mutator in mutators:
                new_model = mutator.bind_sampler(sampler).apply(new_model)
            result = self._get_converted_pytorch_model(new_model)(torch.zeros(1, 1)).item()
            self.assertIn(result, [0., 3., 4., 5.])

        # independent layer choice
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Repeat(lambda index: nn.LayerChoice([AddOne(), nn.Identity()]), (2, 3), label='rep')

            def forward(self, x):
                return self.block(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 4)

        result = []
        for _ in range(20):
            new_model = model
            for mutator in mutators:
                new_model = mutator.bind_sampler(sampler).apply(new_model)
            result.append(self._get_converted_pytorch_model(new_model)(torch.zeros(1, 1)).item())

        self.assertIn(1., result)

    def test_cell(self):
        @model_wrapper
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

        @model_wrapper
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

    def test_cell_predecessors(self):
        from typing import List, Tuple

        class Preprocessor(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 16)

            def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
                return [self.linear(x[0]), x[1]]

        class Postprocessor(nn.Module):
            def forward(self, this: torch.Tensor, prev: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
                return prev[-1], this

        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.Cell({
                    'first': nn.Linear(16, 16),
                    'second': nn.Linear(16, 16, bias=False)
                }, num_nodes=4, num_ops_per_node=2, num_predecessors=2,
                preprocessor=Preprocessor(), postprocessor=Postprocessor(), merge_op='all')

            def forward(self, x, y):
                return self.cell([x, y])

        raw_model, mutators = self._get_model_with_mutators(Net())
        for _ in range(10):
            sampler = EnumerateSampler()
            model = raw_model
            for mutator in mutators:
                model = mutator.bind_sampler(sampler).apply(model)
            result = self._get_converted_pytorch_model(model)(
                torch.randn(1, 3), torch.randn(1, 16))
            self.assertTrue(result[0].size() == torch.Size([1, 16]))
            self.assertTrue(result[1].size() == torch.Size([1, 64]))

    def test_nasbench201_cell(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.NasBench201Cell([
                    lambda x, y: nn.Linear(x, y),
                    lambda x, y: nn.Linear(x, y, bias=False)
                ], 10, 16)

            def forward(self, x):
                return self.cell(x)

        raw_model, mutators = self._get_model_with_mutators(Net())
        for _ in range(10):
            sampler = EnumerateSampler()
            model = raw_model
            for mutator in mutators:
                model = mutator.bind_sampler(sampler).apply(model)
            self.assertTrue(self._get_converted_pytorch_model(model)(torch.randn(2, 10)).size() == torch.Size([2, 16]))

    def test_autoactivation(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.AutoActivation()

            def forward(self, x):
                return self.act(x)

        raw_model, mutators = self._get_model_with_mutators(Net())
        for _ in range(10):
            sampler = EnumerateSampler()
            model = raw_model
            for mutator in mutators:
                model = mutator.bind_sampler(sampler).apply(model)
            self.assertTrue(self._get_converted_pytorch_model(model)(torch.randn(2, 10)).size() == torch.Size([2, 10]))


class Python(GraphIR):
    def _get_converted_pytorch_model(self, model_ir):
        mutation = {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model_ir.history}
        with ContextStack('fixed', mutation):
            model = model_ir.python_class(**model_ir.python_init_params)
            return model

    def _get_model_with_mutators(self, pytorch_model):
        return extract_mutation_from_pt_module(pytorch_model)

    @unittest.skip
    def test_value_choice(self): ...

    @unittest.skip
    def test_value_choice_in_functional(self): ...

    @unittest.skip
    def test_valuechoice_access_functional(self): ...

    @unittest.skip
    def test_valuechoice_access_functional_expression(self): ...

    def test_cell_loose_end(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                                    num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='loose_end')

            def forward(self, x, y):
                return self.cell([x, y])

        raw_model, mutators = self._get_model_with_mutators(Net())
        any_not_all = False
        for _ in range(10):
            sampler = EnumerateSampler()
            model = raw_model
            for mutator in mutators:
                model = mutator.bind_sampler(sampler).apply(model)
            model = self._get_converted_pytorch_model(model)
            indices = model.cell.output_node_indices
            assert all(i > 2 for i in indices)
            self.assertTrue(model(torch.randn(1, 16), torch.randn(1, 16)).size() == torch.Size([1, 16 * len(indices)]))
            if len(indices) < 4:
                any_not_all = True
        self.assertTrue(any_not_all)

    def test_cell_complex(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.Cell({
                    'first': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16),
                    'second': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16, bias=False)
                }, num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

            def forward(self, x, y):
                return self.cell([x, y])

        raw_model, mutators = self._get_model_with_mutators(Net())
        for _ in range(10):
            sampler = EnumerateSampler()
            model = raw_model
            for mutator in mutators:
                model = mutator.bind_sampler(sampler).apply(model)
            self.assertTrue(self._get_converted_pytorch_model(model)(
                torch.randn(1, 3), torch.randn(1, 16)).size() == torch.Size([1, 64]))

    def test_nasbench101_cell(self):
        # this is only supported in python engine for now.
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.NasBench101Cell([lambda x: nn.Linear(x, x), lambda x: nn.Linear(x, x, bias=False)],
                                               10, 16, lambda x, y: nn.Linear(x, y), max_num_nodes=5, max_num_edges=7)

            def forward(self, x):
                return self.cell(x)

        raw_model, mutators = self._get_model_with_mutators(Net())

        succeeded = 0
        sampler = RandomSampler()
        while succeeded <= 10:
            try:
                model = raw_model
                for mutator in mutators:
                    model = mutator.bind_sampler(sampler).apply(model)
                succeeded += 1
            except InvalidMutation:
                continue
            self.assertTrue(self._get_converted_pytorch_model(model)(torch.randn(2, 10)).size() == torch.Size([2, 16]))


class Shared(unittest.TestCase):
    # This kind of tests are general across execution engines

    def test_valuechoice_in_evaluator(self):
        def foo():
            pass

        evaluator = FunctionalEvaluator(foo, t=1, x=2)
        assert process_evaluator_mutations(evaluator, []) == []

        evaluator = FunctionalEvaluator(foo, t=1, x=ValueChoice([1, 2]), y=ValueChoice([3, 4]))
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 2
        init_model = Model(_internal=True)
        init_model.evaluator = evaluator
        sampler = EnumerateSampler()
        model = mutators[0].bind_sampler(sampler).apply(init_model)
        assert model.evaluator.trace_kwargs['x'] == 1
        model = mutators[0].bind_sampler(sampler).apply(init_model)
        assert model.evaluator.trace_kwargs['x'] == 2

        # share label
        evaluator = FunctionalEvaluator(foo, t=ValueChoice([1, 2], label='x'), x=ValueChoice([1, 2], label='x'))
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 1

        # getitem
        choice = ValueChoice([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        evaluator = FunctionalEvaluator(foo, t=1, x=choice['a'], y=choice['b'])
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 1
        init_model = Model(_internal=True)
        init_model.evaluator = evaluator
        sampler = RandomSampler()
        for _ in range(10):
            model = mutators[0].bind_sampler(sampler).apply(init_model)
            assert (model.evaluator.trace_kwargs['x'], model.evaluator.trace_kwargs['y']) in [(1, 2), (3, 4)]
