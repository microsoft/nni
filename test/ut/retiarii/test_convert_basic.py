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

    def checkExportImport(self, model, input, check_value=True):
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
        #print('check zql: ', expected_output, converted_output)
        if check_value:
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

    def test_basic_new_empty(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.new_empty((2, 3), dtype=torch.int8, device=torch.device('cpu'))
                return out
        self.checkExportImport(SimpleOp(), (torch.ones(()), ), check_value=False)

    # skip torch.Tensor.new_ones as it is not supported by jit

    def test_basic_new_zeros(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.new_zeros((2, 3))
                return out
        self.checkExportImport(SimpleOp(), (torch.tensor((), dtype=torch.int32), ))

    def test_basic_is_cuda(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return torch.tensor([x.is_cuda], dtype=torch.bool, device=torch.device('cpu'))
        self.checkExportImport(SimpleOp(), (torch.tensor((), dtype=torch.int32), ), check_value=False)

    # is_quantized
    # is_meta
    # device
    # grad
    # ndim
    # T
    # real
    # imag

    def test_basic_abs(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out1 = x.abs()
                out11 = x.absolute()
                out2 = torch.abs(x)
                out3 = x.abs_()
                out33 = x.absolute_()
                return out1, out11, out2, out3, out33
        self.checkExportImport(SimpleOp(), (torch.tensor([-1, -2, 3]), ))

    # TODO: topological sort should be improved
    #def forward(self, x__1):
    #    __Acos2 = x__1.acos()
    #    __Acos_3 = x__1.acos_()
    #    __Acos1 = x__1.acos()
    #    __TupleConstruct4 = (__Acos1,__Acos2,__Acos_3)
    #    return __TupleConstruct4
    def test_basic_acos(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out1 = x.acos()
                out2 = torch.acos(x)
                # TODO: add back this line
                #out3 = x.acos_()
                return out1, out2#, out3
        self.checkExportImport(SimpleOp(), (torch.tensor([-1.0, -0.5, 0.2]), ))

    # arccos is not supported by jit

    def test_basic_add(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                t = torch.tensor([-1.0, -0.5, 0.2])
                out1 = x.add(t)
                out2 = x.add(t, alpha=2)
                #out3 = x.add_(t)
                return out1, out2#, out3
        self.checkExportImport(SimpleOp(), (torch.tensor([-1.0, -0.5, 0.2]), ))

    def test_basic_addbmm(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y, z):
                out1 = x.addbmm(y, z, beta=2, alpha=3)
                out2 = torch.addbmm(x, y, z, beta=2, alpha=3)
                #out3 = x.addbmm_(y, z, beta=2, alpha=3)
                return out1, out2#, out3
        self.checkExportImport(SimpleOp(), (torch.randn(3, 5), torch.randn(10, 3, 4), torch.randn(10, 4, 5), ))

    def test_basic_addcdiv(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y, z):
                out1 = x.addcdiv(y, z, value=2)
                out2 = torch.addcdiv(x, y, z, value=2)
                # addcdiv_
                return out1, out2
        self.checkExportImport(SimpleOp(), (torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), ))

    def test_basic_addcmul(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y, z):
                out1 = x.addcmul(y, z, value=0.1)
                out2 = torch.addcmul(x, y, z, value=0.1)
                # addcmul_
                return out1, out2
        self.checkExportImport(SimpleOp(), (torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), ))

    def test_basic_addmm(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y, z):
                out1 = x.addmm(y, z, beta=0.1, alpha=0.2)
                out2 = torch.addmm(x, y, z, beta=0.1, alpha=0.2)
                # addmm_
                return out1, out2
        self.checkExportImport(SimpleOp(), (torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3), ))

    def test_basic_addmv(self):
        class SimpleOp(nn.Module):
            def forward(self, x, y, z):
                out1 = x.addmv(y, z, beta=0.1, alpha=0.2)
                out2 = torch.addmv(x, y, z, beta=0.1, alpha=0.2)
                return out1, out2
        self.checkExportImport(SimpleOp(), (torch.randn(2), torch.randn(2, 3), torch.randn(3), ))
        
    '''def test_basic_(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return
        self.checkExportImport(SimpleOp(), )
        
    def test_basic_(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return
        self.checkExportImport(SimpleOp(), )
        
    def test_basic_(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return
        self.checkExportImport(SimpleOp(), )
        
    def test_basic_(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return
        self.checkExportImport(SimpleOp(), )
        
    def test_basic_(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return
        self.checkExportImport(SimpleOp(), )
        
    def test_basic_(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return
        self.checkExportImport(SimpleOp(), )
        
    def test_basic_(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                return
        self.checkExportImport(SimpleOp(), )'''