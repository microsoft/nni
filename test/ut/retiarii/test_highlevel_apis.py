import unittest

import nni.retiarii.nn.pytorch as nn
import torch
from nni.retiarii import Sampler, blackbox_module
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.nn.pytorch.mutator import process_inline_mutation


class EnuemrateSampler(Sampler):
    def __init__(self):
        self.index = 0

    def choice(self, candidates, *args, **kwargs):
        choice = candidates[self.index % len(candidates)]
        self.index += 1
        return choice


@blackbox_module
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


class TestHighLevelAPI(unittest.TestCase):

    def _convert_to_ir(self, model):
        script_module = torch.jit.script(model)
        return convert_to_graph(script_module, model)

    def _get_converted_pytorch_model(self, model_ir):
        model_code = model_to_pytorch_script(model_ir)
        exec_vars = {}
        exec(model_code + '\n\nconverted_model = _model()', exec_vars)
        return exec_vars['converted_model']

    def test_layer_choice(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.LayerChoice([
                    nn.Conv2d(3, 3, kernel_size=1),
                    nn.Conv2d(3, 5, kernel_size=1)
                ])

            def forward(self, x):
                return self.module(x)

        model = self._convert_to_ir(Net())
        mutators = process_inline_mutation(model)
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnuemrateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 3, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 5, 3, 3]))

    def test_input_choice(self):
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

        model = self._convert_to_ir(Net())
        mutators = process_inline_mutation(model)
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnuemrateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 3, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 5, 3, 3]))

    def test_value_choice(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.index = nn.ValueChoice([0, 1])
                self.conv = MutableConv()

            def forward(self, x):
                return self.conv(x, self.index())

        model = self._convert_to_ir(Net())
        mutators = process_inline_mutation(model)
        self.assertEqual(len(mutators), 1)
        mutator = mutators[0].bind_sampler(EnuemrateSampler())
        model1 = mutator.apply(model)
        model2 = mutator.apply(model)
        self.assertEqual(self._get_converted_pytorch_model(model1)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 3, 3, 3]))
        self.assertEqual(self._get_converted_pytorch_model(model2)(torch.randn(1, 3, 3, 3)).size(),
                         torch.Size([1, 5, 3, 3]))

