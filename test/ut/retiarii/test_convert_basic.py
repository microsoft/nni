import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import blackbox_module
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.utils import get_records

class TestConvert(unittest.TestCase):
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

    def checkExportImport(self, model, input):
        script_module = torch.jit.script(model)
        model_ir = convert_to_graph(script_module, model)
        model_code = model_to_pytorch_script(model_ir)
        #print('zql: ', model_code)
        #exit(1)
        from nni.retiarii.nn.pytorch.inject_nn import remove_inject_pytorch_nn
        remove_inject_pytorch_nn()

        exec_vars = {}
        exec(model_code + '\n\nconverted_model = _model()', exec_vars)
        converted_model = exec_vars['converted_model']
        converted_state_dict = self._match_state_dict(list(model.state_dict().values()),
                                                      dict(converted_model.state_dict()))
        converted_model.load_state_dict(converted_state_dict)
        with torch.no_grad():
            expected_output = model.eval()(*input)
            converted_output = converted_model.eval()(*input)
        self.assertEqual(len(converted_output), len(expected_output))
        for a, b in zip(converted_output, expected_output):
            self.assertLess((a - b).abs().max().item(), 1E-4)
        return converted_model

    # skip torch.Tensor.new_tensor as it is not supported by jit

    def test_basic_new_full(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                # requires_grad is not supported by jit
                # aten::new_full(Tensor self, int[] size, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor):
                # Keyword argument requires_grad unknown.
                out = x.new_full((3, 4), 3.141592, dtype=torch.float32, device=torch.device('cpu'))
                return out
        self.checkExportImport(SimpleOp(), (torch.ones((2,), dtype=torch.float64), ))