import math
import random
import unittest
from collections import Counter

import pytest

import nni
import nni.retiarii.evaluator.pytorch.lightning as pl
import nni.retiarii.nn.pytorch as nn
import pytorch_lightning
import torch
import torch.nn.functional as F
from nni.retiarii import InvalidMutation, Sampler, basic_unit
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.execution.utils import _unpack_if_only_one
from nni.retiarii.experiment.pytorch import preprocess_model
from nni.retiarii.graph import Model
from nni.retiarii.nn.pytorch.api import ValueChoice
from nni.retiarii.nn.pytorch.mutator import process_evaluator_mutations, process_inline_mutation, extract_mutation_from_pt_module
from nni.retiarii.serializer import model_wrapper
from nni.retiarii.utils import ContextStack, NoContextError, original_state_dict_hooks


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


def _apply_all_mutators(model, mutators, samplers):
    if not isinstance(samplers, list):
        samplers = [samplers for _ in range(len(mutators))]
    assert len(samplers) == len(mutators)
    model_new = model
    for mutator, sampler in zip(mutators, samplers):
        model_new = mutator.bind_sampler(sampler).apply(model_new)
    return model_new


class GraphIR(unittest.TestCase):
    # graph engine will have an extra mutator for parameter choices
    value_choice_incr = 1
    # graph engine has an extra mutator to apply the depth choice to nodes
    repeat_incr = 1
    # graph engine parse the model into graph
    graph_engine = True

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

    def test_layer_choice_weight_inheritance(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.LayerChoice([nn.Conv2d(3, i, kernel_size=1) for i in range(1, 11)])

            def forward(self, x):
                return self.module(x)

        orig_model = Net()
        model, mutators = self._get_model_with_mutators(orig_model)
        mutator = mutators[0].bind_sampler(EnumerateSampler())
        for i in range(1, 11):
            model_new = mutator.apply(model)
            model_new = self._get_converted_pytorch_model(model_new)
            with original_state_dict_hooks(model_new):
                model_new.load_state_dict(orig_model.state_dict(), strict=False)
            inp = torch.randn(1, 3, 3, 3)
            a = getattr(orig_model.module, str(i - 1))(inp)
            b = model_new(inp)
            self.assertLess((a - b).abs().max().item(), 1E-4)

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

    def test_nested_layer_choice_weight_inheritance(self):
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

        orig_model = Net()
        model, mutators = self._get_model_with_mutators(orig_model)
        mutators[0].bind_sampler(EnumerateSampler())
        mutators[1].bind_sampler(EnumerateSampler())
        input = torch.randn(1, 3, 5, 5)

        for i in range(3):
            model_new = self._get_converted_pytorch_model(mutators[1].apply(mutators[0].apply(model)))
            with original_state_dict_hooks(model_new):
                model_new.load_state_dict(orig_model.state_dict(), strict=False)
            if i == 0:
                a = getattr(getattr(orig_model.module, '0'), '0')(input)
            elif i == 1:
                a = getattr(orig_model.module, '1')(input)
            elif i == 2:
                a = getattr(getattr(orig_model.module, '0'), '2')(input)
            b = model_new(input)
            self.assertLess((a - b).abs().max().item(), 1E-4)

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
        self.assertEqual(len(mutators), 1 + self.value_choice_incr)
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
        self.assertEqual(len(mutators), self.value_choice_incr + 1)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        model1 = _apply_all_mutators(model, mutators, samplers)
        model2 = _apply_all_mutators(model, mutators, samplers)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 1, 1]))

    def test_value_choice_as_two_parameters(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, nn.ValueChoice([6, 8]), kernel_size=nn.ValueChoice([3, 5]))

            def forward(self, x):
                return self.conv(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 2 + self.value_choice_incr)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        model1 = _apply_all_mutators(model, mutators, samplers)
        model2 = _apply_all_mutators(model, mutators, samplers)
        input = torch.randn(1, 3, 5, 5)
        self.assertEqual(self._get_converted_pytorch_model(model1)(input).size(),
                         torch.Size([1, 6, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(input).size(),
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
        self.assertEqual(len(mutators), 1 + self.value_choice_incr)
        sampler = EnumerateSampler()
        model1 = _apply_all_mutators(model, mutators, sampler)
        model2 = _apply_all_mutators(model, mutators, sampler)
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
        self.assertEqual(len(mutators), 3 + self.value_choice_incr)
        sz_counter = Counter()
        sampler = RandomSampler()
        for i in range(100):
            model_new = _apply_all_mutators(model, mutators, sampler)
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

    def test_valuechoice_getitem(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                vc = nn.ValueChoice([(6, 3), (8, 5)])
                self.conv = nn.Conv2d(3, vc[0], kernel_size=vc[1])

            def forward(self, x):
                return self.conv(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1 + self.value_choice_incr)
        sampler = EnumerateSampler()
        input = torch.randn(1, 3, 5, 5)
        self.assertEqual(self._get_converted_pytorch_model(_apply_all_mutators(model, mutators, sampler))(input).size(),
                         torch.Size([1, 6, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(_apply_all_mutators(model, mutators, sampler))(input).size(),
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
        self.assertEqual(len(mutators), 1 + self.value_choice_incr)
        input = torch.randn(1, 3, 5, 5)
        self._get_converted_pytorch_model(_apply_all_mutators(model, mutators, EnumerateSampler()))(input)

    def test_valuechoice_getitem_functional(self):
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

    def test_valuechoice_getitem_functional_expression(self):
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

    def test_valuechoice_multi(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                choice1 = nn.ValueChoice([{"in": 1, "out": 3}, {"in": 2, "out": 6}, {"in": 3, "out": 9}])
                choice2 = nn.ValueChoice([2.5, 3.0, 3.5], label='multi')
                choice3 = nn.ValueChoice([2.5, 3.0, 3.5], label='multi')
                self.conv1 = nn.Conv2d(choice1["in"], round(choice1["out"] * choice2), 1)
                self.conv2 = nn.Conv2d(choice1["in"], round(choice1["out"] * choice3), 1)

            def forward(self, x):
                return self.conv1(x) + self.conv2(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 2 + self.value_choice_incr)
        samplers = [EnumerateSampler()] + [RandomSampler() for _ in range(self.value_choice_incr + 1)]

        for i in range(10):
            model_new = _apply_all_mutators(model, mutators, samplers)
            result = self._get_converted_pytorch_model(model_new)(torch.randn(1, i % 3 + 1, 3, 3))
            self.assertIn(result.size(), [torch.Size([1, round((i % 3 + 1) * 3 * k), 3, 3]) for k in [2.5, 3.0, 3.5]])

    def test_valuechoice_inconsistent_label(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, nn.ValueChoice([3, 5], label='a'), 1)
                self.conv2 = nn.Conv2d(3, nn.ValueChoice([3, 6], label='a'), 1)

            def forward(self, x):
                return torch.cat([self.conv1(x), self.conv2(x)], 1)

        with pytest.raises(AssertionError):
            self._get_model_with_mutators(Net())

    def test_valuechoice_hybrid_arch_hparams(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 5, kernel_size=nn.ValueChoice([3, 5]))

            def forward(self, x):
                return self.conv(x)

        def foo():
            pass

        evaluator = FunctionalEvaluator(foo, t=1, x=ValueChoice([1, 2]), y=ValueChoice([3, 4]))
        model, mutators = preprocess_model(Net(), evaluator, [], full_ir=self.graph_engine)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        model1 = _apply_all_mutators(model, mutators, samplers)
        model2 = _apply_all_mutators(model, mutators, samplers)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 3, 3]))
        self.assertEqual(model1.evaluator.trace_kwargs['x'], 1)
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 5, 5)).size(),
                         torch.Size([1, 5, 1, 1]))
        self.assertEqual(model2.evaluator.trace_kwargs['y'], 4)

    def test_valuechoice_hybrid_arch_hparams_conflict_label(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 5, kernel_size=nn.ValueChoice([3, 5], label='123'))

            def forward(self, x):
                return self.conv(x)

        def foo():
            pass

        evaluator = FunctionalEvaluator(foo, t=1, x=ValueChoice([3, 5], label='123'))
        with pytest.raises(ValueError, match='share'):
            preprocess_model(Net(), evaluator, [], full_ir=self.graph_engine)

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
        self.assertEqual(len(mutators), 1 + self.repeat_incr + self.value_choice_incr)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        for target in [3, 4, 5]:
            new_model = _apply_all_mutators(model, mutators, samplers)
            self.assertTrue((self._get_converted_pytorch_model(new_model)(torch.zeros(1, 16)) == target).all())

    def test_repeat_static(self):
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1

        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Repeat(lambda index: nn.LayerChoice([AddOne(), nn.Identity()]), 4)

            def forward(self, x):
                return self.block(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 4)
        sampler = RandomSampler()

        result = []
        for _ in range(50):
            new_model = model
            for mutator in mutators:
                new_model = mutator.bind_sampler(sampler).apply(new_model)
            result.append(self._get_converted_pytorch_model(new_model)(torch.zeros(1, 1)).item())

        for x in [1, 2, 3]:
            self.assertIn(float(x), result)

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
        self.assertEqual(len(mutators), 2 + self.repeat_incr + self.value_choice_incr)
        self.assertEqual(set([mutator.label for mutator in mutators if mutator.label is not None]), {'lc', 'rep'})

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
        self.assertEqual(len(mutators), 4 + self.repeat_incr + self.value_choice_incr)

        result = []
        for _ in range(20):
            new_model = model
            for mutator in mutators:
                new_model = mutator.bind_sampler(sampler).apply(new_model)
            result.append(self._get_converted_pytorch_model(new_model)(torch.zeros(1, 1)).item())

        self.assertIn(1., result)

    def test_repeat_valuechoice(self):
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1

        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Repeat(AddOne(), nn.ValueChoice([1, 3, 5]))

            def forward(self, x):
                return self.block(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1 + self.repeat_incr + self.value_choice_incr)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        for target in [1, 3, 5]:
            new_model = _apply_all_mutators(model, mutators, samplers)
            self.assertTrue((self._get_converted_pytorch_model(new_model)(torch.zeros(1, 16)) == target).all())

    def test_repeat_valuechoicex(self):
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1

        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Repeat(AddOne(), nn.ValueChoice([0, 2, 4]) + 1)

            def forward(self, x):
                return self.block(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1 + self.repeat_incr + self.value_choice_incr)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        for target in [1, 3, 5]:
            new_model = _apply_all_mutators(model, mutators, samplers)
            self.assertTrue((self._get_converted_pytorch_model(new_model)(torch.zeros(1, 16)) == target).all())

    def test_repeat_weight_inheritance(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.Repeat(lambda index: nn.Conv2d(3, 3, 1), (2, 5))

            def forward(self, x):
                return self.module(x)

        orig_model = Net()
        model, mutators = self._get_model_with_mutators(orig_model)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        inp = torch.randn(1, 3, 5, 5)

        for i in range(4):
            model_new = self._get_converted_pytorch_model(_apply_all_mutators(model, mutators, samplers))
            with original_state_dict_hooks(model_new):
                model_new.load_state_dict(orig_model.state_dict(), strict=False)

            a = nn.Sequential(*orig_model.module.blocks[:i + 2])(inp)
            b = model_new(inp)
            self.assertLess((a - b).abs().max().item(), 1E-4)

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
    # Python engine doesn't have the extra mutator
    value_choice_incr = 0
    repeat_incr = 0
    graph_engine = False

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
    def test_valuechoice_getitem_functional(self): ...

    @unittest.skip
    def test_valuechoice_getitem_functional_expression(self): ...

    def test_repeat_zero(self):
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1

        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Repeat(AddOne(), (0, 3))

            def forward(self, x):
                return self.block(x)

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1 + self.repeat_incr + self.value_choice_incr)
        samplers = [EnumerateSampler() for _ in range(len(mutators))]
        for target in [0, 1, 2, 3]:
            new_model = _apply_all_mutators(model, mutators, samplers)
            self.assertTrue((self._get_converted_pytorch_model(new_model)(torch.zeros(1, 16)) == target).all())

    def test_hyperparameter_choice(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.aux = nn.ModelParameterChoice([False, True])

            def forward(self, x):
                return x

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        sampler = EnumerateSampler()
        model1 = _apply_all_mutators(model, mutators, sampler)
        model2 = _apply_all_mutators(model, mutators, sampler)
        self.assertEqual(self._get_converted_pytorch_model(model1).aux, False)
        self.assertEqual(self._get_converted_pytorch_model(model2).aux, True)

    def test_hyperparameter_choice_parameter(self):
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.aux = torch.nn.Parameter(
                    torch.zeros(1, nn.ModelParameterChoice([64, 128, 256], label='a'), 3, 3)
                )

            def forward(self):
                return self.aux
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.choice = nn.ModelParameterChoice([64, 128, 256], label='a')
                self.inner = Inner()

            def forward(self):
                param = self.inner()
                assert param.size(1) == self.choice
                return param

        model, mutators = self._get_model_with_mutators(Net())
        self.assertEqual(len(mutators), 1)
        sampler = RandomSampler()
        result_pool = set()
        for _ in range(20):
            model = _apply_all_mutators(model, mutators, sampler)
            result = self._get_converted_pytorch_model(model)()
            result_pool.add(result.size(1))
        self.assertSetEqual(result_pool, {64, 128, 256})

    def test_hyperparameter_choice_no_model_wrapper(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.choice = nn.ModelParameterChoice([64, 128, 256], label='a')

        with self.assertRaises(NoContextError):
            model = Net()

    def test_cell(self):
        @model_wrapper
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = nn.Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                                    num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

            def forward(self, x, y):
                return self.cell(x, y)

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
                return self.cell(x)

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

    def test_value_choice_api_purely(self):
        a = nn.ValueChoice([1, 2], label='a')
        b = nn.ValueChoice([3, 4], label='b')
        c = nn.ValueChoice([5, 6], label='c')
        d = a + b + 3 * c
        for i, choice in enumerate(d.inner_choices()):
            if i == 0:
                assert choice.candidates == [1, 2]
            elif i == 1:
                assert choice.candidates == [3, 4]
            elif i == 2:
                assert choice.candidates == [5, 6]
        assert d.evaluate([2, 3, 5]) == 20
        expect = [x + y + 3 * z for x in [1, 2] for y in [3, 4] for z in [5, 6]]
        assert list(d.all_options()) == expect

        a = nn.ValueChoice(['cat', 'dog'])
        b = nn.ValueChoice(['milk', 'coffee'])
        assert (a + b).evaluate(['dog', 'coffee']) == 'dogcoffee'
        assert (a + 2 * b).evaluate(['cat', 'milk']) == 'catmilkmilk'

        assert (3 - nn.ValueChoice([1, 2])).evaluate([1]) == 2

        with pytest.raises(TypeError):
            a + nn.ValueChoice([1, 3])

        a = nn.ValueChoice([1, 17])
        a = (abs(-a * 3) % 11) ** 5
        assert 'abs' in repr(a)
        with pytest.raises(ValueError):
            a.evaluate([42])
        assert a.evaluate([17]) == 7 ** 5

        a = round(7 / nn.ValueChoice([2, 5]))
        assert a.evaluate([2]) == 4

        a = ~(77 ^ (nn.ValueChoice([1, 4]) & 5))
        assert a.evaluate([4]) == ~(77 ^ (4 & 5))

        a = nn.ValueChoice([5, 3]) * nn.ValueChoice([6.5, 7.5])
        assert math.floor(a.evaluate([5, 7.5])) == int(5 * 7.5)

        a = nn.ValueChoice([1, 3])
        b = nn.ValueChoice([2, 4])
        with pytest.raises(RuntimeError):
            min(a, b)
        with pytest.raises(RuntimeError):
            if a < b:
                ...

        assert nn.ValueChoice.min(a, b).evaluate([3, 2]) == 2
        assert nn.ValueChoice.max(a, b).evaluate([3, 2]) == 3
        assert nn.ValueChoice.max(1, 2, 3) == 3
        assert nn.ValueChoice.max([1, 3, 2]) == 3

        assert nn.ValueChoice.condition(nn.ValueChoice([2, 3]) <= 2, 'a', 'b').evaluate([3]) == 'b'
        assert nn.ValueChoice.condition(nn.ValueChoice([2, 3]) <= 2, 'a', 'b').evaluate([2]) == 'a'

        with pytest.raises(RuntimeError):
            assert int(nn.ValueChoice([2.5, 3.5])).evalute([2.5]) == 2

        assert nn.ValueChoice.to_int(nn.ValueChoice([2.5, 3.5])).evaluate([2.5]) == 2
        assert nn.ValueChoice.to_float(nn.ValueChoice(['2.5', '3.5'])).evaluate(['3.5']) == 3.5

    def test_make_divisible(self):
        def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
            if min_value is None:
                min_value = divisor
            new_value = nn.ValueChoice.max(min_value, nn.ValueChoice.to_int(value + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than (1-min_ratio).
            return nn.ValueChoice.condition(new_value < min_ratio * value, new_value + divisor, new_value)

        def original_make_divisible(value, divisor, min_value=None, min_ratio=0.9):
            if min_value is None:
                min_value = divisor
            new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than (1-min_ratio).
            if new_value < min_ratio * value:
                new_value += divisor
            return new_value

        values = [4, 8, 16, 32, 64, 128]
        divisors = [2, 3, 5, 7, 15]
        with pytest.raises(RuntimeError):
            original_make_divisible(nn.ValueChoice(values, label='value'), nn.ValueChoice(divisors, label='divisor'))
        result = make_divisible(nn.ValueChoice(values, label='value'), nn.ValueChoice(divisors, label='divisor'))
        for value in values:
            for divisor in divisors:
                lst = [value if choice.label == 'value' else divisor for choice in result.inner_choices()]
                assert result.evaluate(lst) == original_make_divisible(value, divisor)

        assert len(list(result.all_options())) == 30
        assert max(result.all_options()) == 135

    def test_valuechoice_in_evaluator(self):
        def foo():
            pass

        evaluator = FunctionalEvaluator(foo, t=1, x=2)
        assert process_evaluator_mutations(evaluator, []) == []

        evaluator = FunctionalEvaluator(foo, t=1, x=ValueChoice([1, 2]), y=ValueChoice([3, 4]))
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 3
        init_model = Model(_internal=True)
        init_model.evaluator = evaluator
        samplers = [EnumerateSampler() for _ in range(3)]
        model = _apply_all_mutators(init_model, mutators, samplers)
        assert model.evaluator.trace_kwargs['x'] == 1
        model = _apply_all_mutators(init_model, mutators, samplers)
        assert model.evaluator.trace_kwargs['x'] == 2

        # share label
        evaluator = FunctionalEvaluator(foo, t=ValueChoice([1, 2], label='x'), x=ValueChoice([1, 2], label='x'))
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 2

        # getitem
        choice = ValueChoice([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        evaluator = FunctionalEvaluator(foo, t=1, x=choice['a'], y=choice['b'])
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 2
        init_model = Model(_internal=True)
        init_model.evaluator = evaluator
        sampler = RandomSampler()
        for _ in range(10):
            model = _apply_all_mutators(init_model, mutators, sampler)
            assert (model.evaluator.trace_kwargs['x'], model.evaluator.trace_kwargs['y']) in [(1, 2), (3, 4)]

    def test_valuechoice_in_evaluator_nested(self):
        @nni.trace
        class FooClass:
            def __init__(self, a):
                self.a = a

        obj = FooClass(ValueChoice([1, 2, 3], label='t'))

        def foo():
            pass

        evaluator = FunctionalEvaluator(foo, t=obj, v=ValueChoice([1, 2, 3], label='t') + ValueChoice([10, 20, 30]))
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 3
        init_model = Model(_internal=True)
        init_model.evaluator = evaluator
        samplers = [RandomSampler() for _ in range(3)]
        for _ in range(10):
            model = _apply_all_mutators(init_model, mutators, samplers)
            a, v = model.evaluator.trace_kwargs['t'].a, model.evaluator.trace_kwargs['v']
            assert v % 10 == a
            assert a in [1, 2, 3]
            assert v // 10 in [1, 2, 3]

    @unittest.skipIf(pytorch_lightning.__version__ < '1.0', 'Legacy PyTorch-lightning not supported')
    def test_valuechoice_lightning(self):
        @nni.trace
        class AnyModule(pl.LightningModule):
            pass

        evaluator = pl.Lightning(AnyModule(), pl.Trainer(max_epochs=nn.ValueChoice([1, 2, 3])))
        mutators = process_evaluator_mutations(evaluator, [])
        assert len(mutators) == 2
        init_model = Model(_internal=True)
        init_model.evaluator = evaluator
        samplers = [RandomSampler() for _ in range(2)]
        values = []
        for _ in range(20):
            model = _apply_all_mutators(init_model, mutators, samplers)
            values.append(model.evaluator.trainer.max_epochs)
            model._dump()

        assert len(set(values)) == 3

    @unittest.skipIf(pytorch_lightning.__version__ < '1.0', 'Legacy PyTorch-lightning not supported')
    def test_valuechoice_classification(self):
        evaluator = pl.Classification(criterion=nn.CrossEntropyLoss)
        process_evaluator_mutations(evaluator, [])

    def test_retiarii_nn_import(self):
        dummy = torch.zeros(1, 16, 32, 24)
        nn.init.uniform_(dummy)

        conv = nn.Conv2d(1, 3, 1)
        param = nn.Parameter(torch.zeros(1, 3, 24, 24))
