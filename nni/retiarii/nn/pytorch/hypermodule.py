# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from nni.retiarii.serializer import basic_unit

from .api import LayerChoice
from .utils import generate_new_label
from ...utils import version_larger_equal

__all__ = ['AutoActivation']

TorchVersion = '1.5.0'

# ============== unary function modules ==============

@basic_unit
class UnaryIdentity(nn.Module):
    def forward(self, x):
        return x

@basic_unit
class UnaryNegative(nn.Module):
    def forward(self, x):
        return -x

@basic_unit
class UnaryAbs(nn.Module):
    def forward(self, x):
        return torch.abs(x)

@basic_unit
class UnarySquare(nn.Module):
    def forward(self, x):
        return torch.square(x)

@basic_unit
class UnaryPow(nn.Module):
    def forward(self, x):
        return torch.pow(x, 3)

@basic_unit
class UnarySqrt(nn.Module):
    def forward(self, x):
        return torch.sqrt(x)

@basic_unit
class UnaryMul(nn.Module):
    def __init__(self):
        super().__init__()
        # element-wise for now, will change to per-channel trainable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)) # pylint: disable=not-callable
    def forward(self, x):
        return x * self.beta

@basic_unit
class UnaryAdd(nn.Module):
    def __init__(self):
        super().__init__()
        # element-wise for now, will change to per-channel trainable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)) # pylint: disable=not-callable
    def forward(self, x):
        return x + self.beta

@basic_unit
class UnaryLogAbs(nn.Module):
    def forward(self, x):
        return torch.log(torch.abs(x) + 1e-7)

@basic_unit
class UnaryExp(nn.Module):
    def forward(self, x):
        return torch.exp(x)

@basic_unit
class UnarySin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

@basic_unit
class UnaryCos(nn.Module):
    def forward(self, x):
        return torch.cos(x)

@basic_unit
class UnarySinh(nn.Module):
    def forward(self, x):
        return torch.sinh(x)

@basic_unit
class UnaryCosh(nn.Module):
    def forward(self, x):
        return torch.cosh(x)

@basic_unit
class UnaryTanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

if not version_larger_equal(torch.__version__, TorchVersion):
    @basic_unit
    class UnaryAsinh(nn.Module):
        def forward(self, x):
            return torch.asinh(x)

@basic_unit
class UnaryAtan(nn.Module):
    def forward(self, x):
        return torch.atan(x)

if not version_larger_equal(torch.__version__, TorchVersion):
    @basic_unit
    class UnarySinc(nn.Module):
        def forward(self, x):
            return torch.sinc(x)

@basic_unit
class UnaryMax(nn.Module):
    def forward(self, x):
        return torch.max(x, torch.zeros_like(x))

@basic_unit
class UnaryMin(nn.Module):
    def forward(self, x):
        return torch.min(x, torch.zeros_like(x))

@basic_unit
class UnarySigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

@basic_unit
class UnaryLogExp(nn.Module):
    def forward(self, x):
        return torch.log(1 + torch.exp(x))

@basic_unit
class UnaryExpSquare(nn.Module):
    def forward(self, x):
        return torch.exp(-torch.square(x))

@basic_unit
class UnaryErf(nn.Module):
    def forward(self, x):
        return torch.erf(x)

unary_modules = ['UnaryIdentity', 'UnaryNegative', 'UnaryAbs', 'UnarySquare', 'UnaryPow',
    'UnarySqrt', 'UnaryMul', 'UnaryAdd', 'UnaryLogAbs', 'UnaryExp', 'UnarySin', 'UnaryCos',
    'UnarySinh', 'UnaryCosh', 'UnaryTanh', 'UnaryAtan', 'UnaryMax',
    'UnaryMin', 'UnarySigmoid', 'UnaryLogExp', 'UnaryExpSquare', 'UnaryErf']

if not version_larger_equal(torch.__version__, TorchVersion):
    unary_modules.append('UnaryAsinh')
    unary_modules.append('UnarySinc')

# ============== binary function modules ==============

@basic_unit
class BinaryAdd(nn.Module):
    def forward(self, x):
        return x[0] + x[1]

@basic_unit
class BinaryMul(nn.Module):
    def forward(self, x):
        return x[0] * x[1]

@basic_unit
class BinaryMinus(nn.Module):
    def forward(self, x):
        return x[0] - x[1]

@basic_unit
class BinaryDivide(nn.Module):
    def forward(self, x):
        return x[0] / (x[1] + 1e-7)

@basic_unit
class BinaryMax(nn.Module):
    def forward(self, x):
        return torch.max(x[0], x[1])

@basic_unit
class BinaryMin(nn.Module):
    def forward(self, x):
        return torch.min(x[0], x[1])

@basic_unit
class BinarySigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x[0]) * x[1]

@basic_unit
class BinaryExpSquare(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)) # pylint: disable=not-callable
    def forward(self, x):
        return torch.exp(-self.beta * torch.square(x[0] - x[1]))

@basic_unit
class BinaryExpAbs(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)) # pylint: disable=not-callable
    def forward(self, x):
        return torch.exp(-self.beta * torch.abs(x[0] - x[1]))

@basic_unit
class BinaryParamAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)) # pylint: disable=not-callable
    def forward(self, x):
        return self.beta * x[0] + (1 - self.beta) * x[1]

binary_modules = ['BinaryAdd', 'BinaryMul', 'BinaryMinus', 'BinaryDivide', 'BinaryMax',
    'BinaryMin', 'BinarySigmoid', 'BinaryExpSquare', 'BinaryExpAbs', 'BinaryParamAdd']


class AutoActivation(nn.Module):
    """
    This module is an implementation of the paper "Searching for Activation Functions"
    (https://arxiv.org/abs/1710.05941).
    NOTE: current `beta` is not per-channel parameter

    Parameters
    ----------
    unit_num : int
        the number of core units
    """
    def __init__(self, unit_num: int = 1, label: str = None):
        super().__init__()
        self._label = generate_new_label(label)
        self.unaries = nn.ModuleList()
        self.binaries = nn.ModuleList()
        self.first_unary = LayerChoice([eval('{}()'.format(unary)) for unary in unary_modules], label = f'{self.label}__unary_0')
        for i in range(unit_num):
            one_unary = LayerChoice([eval('{}()'.format(unary)) for unary in unary_modules], label = f'{self.label}__unary_{i+1}')
            self.unaries.append(one_unary)
        for i in range(unit_num):
            one_binary = LayerChoice([eval('{}()'.format(binary)) for binary in binary_modules], label = f'{self.label}__binary_{i}')
            self.binaries.append(one_binary)

    @property
    def label(self):
        return self._label

    def forward(self, x):
        out = self.first_unary(x)
        for unary, binary in zip(self.unaries, self.binaries):
            out = binary(torch.stack([out, unary(x)]))
        return out
