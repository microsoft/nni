
'''
The tests in this file is copied and transformed from
`https://github.com/pytorch/pytorch/blob/master/test/onnx/test_operators.py`
'''

import unittest
from typing import (Dict)

import torch

import nni.nas.nn.pytorch.layers as nn

from .convert_mixin import ConvertMixin, ConvertWithShapeMixin

# following pytorch v1.7.1


class TestOperators(unittest.TestCase, ConvertMixin):

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

    @unittest.skip('cannot be parsed by jit')
    def test_basic_concat2(self):
        class SimpleOp(nn.Module):
            def forward(self, inputs):
                out = torch.cat(inputs, 1)
                return out
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.checkExportImport(SimpleOp(), ((x, y), ))


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

    @unittest.skip('cannot be parsed by jit')
    def test_basic_hardtanh(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.nn.Hardtanh(-0.5, 0.5)(x)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


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

    @unittest.skip('No longer works for pytorch 2.0')
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

    @unittest.skip('Removed by PyTorch')
    def test_basic_norm_p1(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.norm(p=1, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    @unittest.skip('Removed by PyTorch')
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
        self.checkExportImport(SimpleOp(), (x, ), check_value=False)


    def test_basic_rand(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.rand(1, 2, 3, 4) + x
                return out
        x = torch.rand(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ), check_value=False)


    def test_basic_empty_like(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ), check_value=False)


    def test_basic_empty_like_opset7(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ), check_value=False)


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

    def test_basic_expand(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.expand(4, 6, 2)
                return out
        x = torch.randn(6, 1, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_ne(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = torch.ne(x, y)
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_reducemax(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.max(x)
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_reducemin(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.min(x)
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_erf(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.erf()
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_dropout(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.max(torch.nn.functional.dropout(x, training=False))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_dropout_default(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.max(torch.nn.functional.dropout(x,))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ), check_value=False)

    def test_basic_dropout_training(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.max(torch.nn.functional.dropout(x))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ), check_value=False)

    def test_basic_nonzero(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.nonzero(x)
                return out
        x = torch.tensor([[[2., 2.], [1., 0.]], [[0., 0.], [1., 1.]]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_gather(self):
        class SimpleOp(nn.Module):
            def forward(self, data, index):
                out = data.gather(1, index)
                return out
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.checkExportImport(SimpleOp(), (data, index, ))


    def test_basic_gather_opset11(self):
        class SimpleOp(nn.Module):
            def forward(self, data, index):
                out = data.gather(1, index)
                return out
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.checkExportImport(SimpleOp(), (data, index, ))


    def test_basic_scatter_add(self):
        class SimpleOp(nn.Module):
            def forward(self, data, indices, values):
                out = data.scatter_add(1, indices, values)
                return out
        data = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.checkExportImport(SimpleOp(), (data, indices, values, ))


    def test_basic_scatter_add_opset11(self):
        class SimpleOp(nn.Module):
            def forward(self, data, indices, values):
                out = data.scatter_add(1, indices, values)
                return out
        data = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.checkExportImport(SimpleOp(), (data, indices, values, ))


    def test_basic_master_opset(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = x + y
                return out
        x = torch.randn(2, 3).float()
        y = torch.randn(2, 3).float()
        self.checkExportImport(SimpleOp(), (x, y, ))


    def test_basic_std(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.std(x, dim=(0, 1), unbiased=True, keepdim=True)
                return out
        x = torch.randn(2, 3, 4).float()
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_cumsum(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.cumsum(x, dim=1)
                return out
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_pixel_shuffle(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.pixel_shuffle(x, upscale_factor=2)
                return out
        x = torch.randn(2, 8, 3, 4).float()
        self.checkExportImport(SimpleOp(), (x, ))

    @unittest.skip('skip as torch.norm is called with prim::CallFunction, also torch.norm is deprecated')
    def test_basic_frobenius_norm(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.norm(x, p="fro", dim=(0, 1), keepdim=True)
                return out
        x = torch.randn(2, 3, 4).float()
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_unfold(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.unfold(dimension=2, size=2, step=2)
                return out
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_remainder(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = torch.remainder(x, y)
                return out
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.checkExportImport(SimpleOp(), (x, y, ))

    def test_basic_fmod(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = torch.fmod(x, y)
                return out
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.checkExportImport(SimpleOp(), (x, y, ))

    @unittest.skip(reason='aten::gelu is not supported')
    def test_basic_gelu(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.nn.functional.gelu(x)
                return out
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    @unittest.skip('skip as it is called with prim::CallFunction, and unknown func definition')
    def test_basic_unique(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.unique(x, dim=0, sorted=True, return_inverse=False, return_counts=True)
                return out
        x = torch.randint(3, (2, 3, 4, 5)).float()
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_meshgrid(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y, z):
                out = torch.meshgrid(x, y, z)
                return out
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y, z, ))


    def test_basic_topk(self):
        class SimpleOp(nn.Module):
            def forward(self, x, k):
                out = torch.topk(x, k)
                return out
        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        self.checkExportImport(SimpleOp(), (x, k, ))


    def test_basic_topk_smallest_unsorted(self):
        class SimpleOp(nn.Module):
            def forward(self, x, k):
                out = torch.topk(x, k, largest=False, sorted=False)
                return out
        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        self.checkExportImport(SimpleOp(), (x, k, ))


    def test_basic_baddbmm(self):
        class SimpleOp(nn.Module):
            def forward(self, x, b1, b2):
                out = torch.baddbmm(x, b1, b2)
                return out
        x = torch.randn(10, 3, 5)
        b1 = torch.randn(10, 3, 4)
        b2 = torch.randn(10, 4, 5)
        self.checkExportImport(SimpleOp(), (x, b1, b2, ))


    def test_basic_round(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.round(x)
                return out
        x = torch.tensor([0.9920, -1.0362, -1.5000, 2.5000], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_dim(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.scalar_tensor(x.dim())
                return out
        x = torch.ones((2, 2), requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    @unittest.skip('Removed by PyTorch')
    def test_basic_det(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.det(x)
                return out
        x = torch.randn(2, 3, 5, 5, device=torch.device('cpu'))
        self.checkExportImport(SimpleOp(), (x, ))

    # the followings are more complex tests

    def test_mm(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y):
                out = torch.mm(x, y)
                return out
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (m1, m2))

    def test_basic_pad(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.ReflectionPad2d((2, 3, 0, 1))

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.tensor([[[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_batchnorm(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_batchnorm_1d(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.BatchNorm1d(2)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.ones(2, 2, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_conv(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Conv2d(16, 13, 3, bias=False)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_conv_onnx_irv4_opset8(self):
        # This test point checks that for opset 8 (or lower), even if
        # keep_initializers_as_inputs is set to False, it is ignored,
        # and initializers are listed as ONNX graph input, in accordance
        # with ONNX IR v3 semantics (which apply to opset version <= 8).
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Conv2d(2, 4, 3, bias=False)
                self.m.weight.data.fill_(1.0)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.ones(1, 2, 5, 7, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_convtranspose(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.ConvTranspose2d(3, 3, 3, stride=3, bias=False,
                                           padding=1, output_padding=2)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.ones(2, 3, 4, 5, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_maxpool(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool1d(3, stride=2)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(20, 16, 50)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_maxpool_dilations(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool1d(2, stride=1, dilation=2)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(20, 16, 50)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_avg_pool2d(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.AvgPool2d(3, stride=2)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(20, 16, 50, 32)
        self.checkExportImport(SimpleOp(), (x, ))

    @unittest.skip('jit error: "Return value was annotated as having type Tensor but is actually of type Tuple[Tensor, Tensor]"')
    def test_basic_maxpool_indices(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool1d(3, stride=2, return_indices=True)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(20, 16, 50)
        self.checkExportImport(SimpleOp(), (x, ))

    @unittest.skip("jit error: Tried to access nonexistent attribute or method 'at' of type '__torch__.test_convert_operators.MyFun'")
    def test_at_op(self):
        from torch.autograd import Function
        x = torch.randn(3, 4)
        class MyFun(Function):
            @staticmethod
            def symbolic(g, x):
                return g.at("add", x, x)
            @staticmethod
            def forward(ctx, x):
                return x + x
        class MyModule(nn.Module):
            def forward(self, x):
                return MyFun.apply(x)
        self.checkExportImport(MyModule(), x)

    def test_basic_logsoftmax(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.LogSoftmax(dim=3)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_elu(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.ELU()

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_selu(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.SELU()

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_upsample_nearest_scale(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.nn.functional.interpolate(x, scale_factor=2.,
                        mode='nearest', recompute_scale_factor=False)
                return out

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_upsample_nearest_scale_default_scale_factor(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.nn.functional.interpolate(x, scale_factor=2.,
                        mode='nearest')
                return out

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_batchnorm_noaffine(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.BatchNorm2d(128, affine=False, momentum=0.3)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(128, 128, 1, 1, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_embedding_bags(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.EmbeddingBag(10, 8)

            def forward(self, x, y):
                out = self.m(x, y)
                return out

        input = torch.tensor([1, 2, 3, 4]).long()
        offset = torch.tensor([0]).long()
        self.checkExportImport(SimpleOp(), (input, offset, ))

    def test_basic_rrelu(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.RReLU()

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_prelu(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.PReLU(2)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_log_sigmoid(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.LogSigmoid()

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x, ))


    def test_basic_linear(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Linear(4, 5, bias=True)

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(3, 4)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_retain_param_name_disabled(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.fc1 = nn.Linear(4, 5, bias=False)
                self.fc1.weight.data.fill_(2.)
                self.fc2 = nn.Linear(5, 6, bias=False)
                self.fc2.weight.data.fill_(3.)
            def forward(self, x):
                return self.fc2(self.fc1(x))

        x = torch.randn(3, 4).float()
        self.checkExportImport(MyModule(), (x, ))

    @unittest.skip('Segmentation fault')
    def test_dict(self):
        class MyModel(nn.Module):
            def forward(self, x_in: Dict):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in[list(x_in.keys())[0]], list(x_in.keys())[0])
                return x_out

        x = {torch.tensor(1.): torch.randn(1, 2, 3)}
        self.checkExportImport(MyModel(), (x, ))

    def test_arange_dynamic(self):
        class TestModel(nn.Module):
            def forward(self, input):
                out = torch.arange(input.shape[0], input.shape[0] + 5, 0.5)
                return out

        input = torch.randn(5, 3, 2)
        self.checkExportImport(TestModel(), (input, ))

    @unittest.skip(reason='"rshift_cpu" not implemented for Float')
    def test_bitshift(self):
        class BitshiftModel(nn.Module):
            def forward(self, input, input2):
                return input >> 1, input2 >> 2

        input = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.checkExportImport(BitshiftModel(), (input, input2, ))

    def test_layer_norm_aten(self):
        class SimpleOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.LayerNorm([10, 10])

            def forward(self, x):
                out = self.m(x)
                return out

        x = torch.randn(20, 5, 10, 10)
        self.checkExportImport(SimpleOp(), (x, ))

    def test_basic_abs(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.abs(x)
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, ))

class TestOperatorsWithShape(TestOperators, ConvertWithShapeMixin):
    pass
