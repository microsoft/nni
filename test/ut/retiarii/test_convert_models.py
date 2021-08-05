import os
import sys
import unittest
from typing import (Dict)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import serialize
from nni.retiarii.codegen import model_to_pytorch_script

from .convert_mixin import ConvertMixin, ConvertWithShapeMixin


class TestModels(unittest.TestCase, ConvertMixin):
    @staticmethod
    def _match_state_dict(current_values, expected_format):
        result = {}
        for k, v in expected_format.items():
            for idx, cv in enumerate(current_values):
                if cv.shape == v.shape:
                    result[k] = cv
                    current_values.pop(idx)
                    break
        return result

    def run_test(self, model, input, check_value=True):
        model_ir = self._convert_model(model, input)
        model_code = model_to_pytorch_script(model_ir)
        print(model_code)

        exec_vars = {}
        exec(model_code + '\n\nconverted_model = _model()', exec_vars)
        converted_model = exec_vars['converted_model']
        converted_state_dict = self._match_state_dict(list(model.state_dict().values()),
                                                      dict(converted_model.state_dict()))
        converted_model.load_state_dict(converted_state_dict)
        with torch.no_grad():
            expected_output = model.eval()(*input)
            converted_output = converted_model.eval()(*input)
        if check_value:
            try:
                self.assertEqual(len(converted_output), len(expected_output))
                for a, b in zip(converted_output, expected_output):
                    torch.eq(a, b)
            except:
                self.assertEqual(converted_output, expected_output)
        return converted_model

    def test_nested_modulelist(self):
        class Net(nn.Module):
            def __init__(self, num_nodes, num_ops_per_node):
                super().__init__()
                self.ops = nn.ModuleList()
                self.num_nodes = num_nodes
                self.num_ops_per_node = num_ops_per_node
                for _ in range(num_nodes):
                    self.ops.append(nn.ModuleList([nn.Linear(16, 16) for __ in range(num_ops_per_node)]))

            def forward(self, x):
                state = x
                for ops in self.ops:
                    for op in ops:
                        state = op(state)
                return state

        model = Net(4, 2)
        x = torch.rand((16, 16), dtype=torch.float)
        self.run_test(model, (x, ))

    def test_append_input_tensor(self):
        from typing import List
        class Net(nn.Module):
            def __init__(self, num_nodes):
                super().__init__()
                self.ops = nn.ModuleList()
                self.num_nodes = num_nodes
                for _ in range(num_nodes):
                    self.ops.append(nn.Linear(16, 16))
            def forward(self, x: List[torch.Tensor]):
                state = x
                for ops in self.ops:
                    state.append(ops(state[-1]))
                return state[-1]

        model = Net(4)
        x = torch.rand((1, 16), dtype=torch.float)
        self.run_test(model, ([x], ))

class TestModelsWithShape(TestModels, ConvertWithShapeMixin):
    pass
