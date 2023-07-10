# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import functools
import torch
import torch.nn.functional as F

from nni.common.concrete_trace_utils import concrete_trace
from nni.compression.pruning import L1NormPruner
from nni.compression.speedup import ModelSpeedup
from nni.compression.utils.counter import count_flops_params

def simple_annotation():
    def simple_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            output = old_func(*args, **kwargs)
            return output
        return new_func
    return simple_wrapper

class WithAnno1(torch.nn.Module):
    """
    test for annotations
    """
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU6()

    @simple_annotation()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(input)

class WithAnno2(torch.nn.Module):
    """
    test for annotations
    """
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU6()

    @simple_annotation()
    @simple_annotation()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(input)

class WithAnno3(torch.nn.Module):
    """
    test for annotations
    """
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU6()

    @simple_annotation()
    @simple_annotation()
    @simple_annotation()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(input)

class TorchModel(torch.nn.Module):
    def __init__(self, relu):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.relu1 = relu
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.relu2 = torch.nn.ReLU6()
        self.relu3 = torch.nn.ReLU6()
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, x.size()[1:].numel())
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class InitMaskTestCase(unittest.TestCase):
    def the_test_with_annotations(self, relu):
        torch.manual_seed(100)
        model = TorchModel(relu)
        dummy_input = torch.rand(3, 1, 28, 28)

        config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.5}]
        pruner = L1NormPruner(model=model, config_list=config_list)
        _, masks = pruner.compress()
        pruner.unwrap_model()  # unwrap all modules to normal state

        masks['relu1'] = {
            '_input_input': torch.ones((8, 20, 24, 24)),
            '_output_0': torch.ones((8, 20, 24, 24)),
        }
        masks['conv1']['_output_0'] = torch.ones((8, 20, 24, 24))

        traced_model = concrete_trace(model, {'x': dummy_input}, leaf_module=(WithAnno1, WithAnno2, WithAnno3))
        ModelSpeedup(traced_model, (dummy_input,), masks).speedup_model()
        traced_model(dummy_input)

        print('model before speedup', repr(model))
        # 125.49 M, 0.85M, 93.29, 1.1012
        flops, params, _ = count_flops_params(model, dummy_input, verbose=False)
        print(f'Pretrained model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M')

        print('model after speedup', repr(traced_model))
        flops, params, _ = count_flops_params(traced_model, dummy_input, verbose=False)
        print(f'Pruned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M')

    def test_with_annotation0(self):
        return self.the_test_with_annotations(torch.nn.ReLU6())

    def test_with_annotation1(self):
        return self.the_test_with_annotations(WithAnno1())

    def test_with_annotation2(self):
        return self.the_test_with_annotations(WithAnno2())

    def test_with_annotation3(self):
        return self.the_test_with_annotations(WithAnno3())

if __name__ == '__main__':
    # unittest.main()
    InitMaskTestCase().test_with_annotation1()
