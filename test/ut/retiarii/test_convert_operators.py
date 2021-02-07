
'''
The tests in this file is copied and transformed from 
`https://github.com/pytorch/pytorch/blob/master/test/onnx/test_operators.py`
'''

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

# following pytorch v1.7.1

class TestOperators(unittest.TestCase):
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

    def checkExportImport(self, model, input, check_value=True):
        script_module = torch.jit.script(model)
        model_ir = convert_to_graph(script_module, model)
        model_code = model_to_pytorch_script(model_ir)

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

    def test_basic_basic(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y, ))

    def test_basic_view(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.view(1, 1)
                return out
        x = torch.tensor([0.0], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_index(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x[0]
                return out
        x = torch.tensor([[0.0]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_type_as(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.type_as(x)
                return out
        x = torch.tensor([0.0], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_addconstant(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x + 1
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_add_broadcast(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y, ))

    def test_basic_add_left_broadcast(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x + y
                return out
        x = torch.randn(3, requires_grad=True).double()
        y = torch.randn(2, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_add_size1_broadcast(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(2, 1, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_add_size1_right_broadcast(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_add_size1_singleton_broadcast(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(1, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_rsub(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = 1 - x
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_transpose(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.transpose(0, 1).transpose(1, 0)
                return out
        x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_chunk(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.chunk(2)
                return out
        x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_split(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.split(x, 2, 1)
                return out
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_split_with_sizes(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.split(x, [2, 1, 3], 1)
                return out
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.checkExportImport(SimpleOp(), (x, ))


    '''def test_basic_concat2(self):
        class SimpleOp(nn.Module):
            def forward(self, inputs):
                out = torch.cat(inputs, 1)
                return out
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.checkExportImport(SimpleOp(), ((x, y), ))'''


    def test_basic_addmm(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y, z):
                out = torch.addmm(torch.addmm(z, x, y), x, y)
                return out
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        m3 = torch.randn(4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (m1, m2, m3, ))


    def test_basic_permute2(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.permute(0, 1, 4, 2, 5, 3)
                return out
        x = torch.tensor([[[[[[0.0]]]]]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_params(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_params_onnx_irv4(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_clip(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.clamp(x, min=-0.5, max=0.5)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_clip_min(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.clamp(min=-0.1)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_clip_max(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.clamp(max=0.1)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    '''def test_basic_hardtanh(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.nn.Hardtanh(-0.5, 0.5)(x)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))'''


    def test_basic_full(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.full(x.shape, 2., dtype=torch.float32, layout=torch.strided, device=torch.device('cpu'))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_full_like(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.full_like(x, 2, memory_format=torch.preserve_format)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_max(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = torch.max(x, y)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_min(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = torch.min(x, y)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_mean(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.mean(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reduced_mean(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.mean(x, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reduced_mean_keepdim(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.mean(x, dim=(2, 3), keepdim=True)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_sum(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.sum(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reduced_sum(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.sum(x, dim=(1, 2))
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reduced_sum_keepdim(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.sum(x, dim=2, keepdim=True)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_prod(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.prod(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reduced_prod(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.prod(x, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reduced_prod_keepdim(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.prod(x, dim=2, keepdim=True)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_sqrt(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.sqrt(x)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_rsqrt(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.rsqrt(x)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_equal(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x == y
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_lt(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x < y
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_gt(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x > y
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_le(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x <= y
                return out
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_ge(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x >= y
                return out
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y, ))

    def test_basic_exp(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.exp()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_sin(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.sin()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_cos(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.cos()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_tan(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.tan()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_asin(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.asin()
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_acos(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.acos()
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_slice(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x[:, 1:2]
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_slice_dynamic(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x[x.size(0):, x.size(1) - 3]
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_sign(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.sign()
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_narrow(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.narrow(x, 0, 0, 2)
                return out
        x = torch.randn(3, 3, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_atan(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.atan()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_view_flatten(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.view(x.size()[0], x.numel() // x.size()[0])
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_flatten(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.flatten(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_flatten2D(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.flatten(x, 1)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_isnan(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.isnan(x)
                return out
        x = torch.tensor([1, float('nan'), 2])
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_argmax(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.argmax(x, dim=1)
                return out
        x = torch.randn(4, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_pow(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x.pow(y)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_repeat(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.repeat(1, 2, 3, 4)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_repeat_dim_overflow(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.repeat(1, 2, 3, 4)
                return out
        x = torch.randn(1, 2, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_norm_p1(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.norm(p=1, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_norm_p2(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.norm(p=2, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_upsample_nearest_size(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.nn.functional.interpolate(x, size=16, mode='nearest')
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_unsqueeze(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.unsqueeze(len(x.shape))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_implicit_expand(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x + 1
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reduce_sum_negative_indices(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.sum(-1)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_randn(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.randn(1, 2, 3, 4) + x
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_rand(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.rand(1, 2, 3, 4) + x
                return out
        x = torch.rand(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_empty_like(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_empty_like_opset7(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_zeros_like(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.zeros_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_ones_like(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.ones_like(x)
                return out
        x = torch.randn(6, 10, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))