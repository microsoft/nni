# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch
import torch.nn as nn

from .api import LayerChoice

__all__ = ['AutoActivation']

class UnaryMul(nn.Module):
    def __init__(self):
        super().__init__()
        # element-wise for now, will change to per-channel trainable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
    def forward(self, x):
        return x * self.beta

class UnaryAdd(nn.Module):
    def __init__(self):
        super().__init__()
        # element-wise for now, will change to per-channel trainable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
    def forward(self, x):
        return x + self.beta

class BinaryExpSquare(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
    def forward(self, x, y):
        return torch.exp(-self.beta * torch.square(x - y))

class BinaryExpAbs(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
    def forward(self, x, y):
        return torch.exp(-self.beta * torch.abs(x - y))

class BinaryAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
    def forward(self, x, y):
        return self.beta * x + (1 - self.beta) * y

autoact_unary_funcs = [
    lambda x: x,
    lambda x: -x,
    lambda x: torch.abs(x),
    lambda x: torch.square(x),
    lambda x: torch.pow(x, 3),
    lambda x: torch.pow(x, 0.5),
    #UnaryMul(),
    #UnaryAdd(),
    lambda x: torch.log(torch.abs(x) + 1e-7),
    lambda x: torch.exp(x),
    lambda x: torch.sin(x),
    lambda x: torch.cos(x),
    lambda x: torch.sinh(x),
    lambda x: torch.cosh(x),
    lambda x: torch.tanh(x),
    lambda x: torch.asinh(x),
    lambda x: torch.atan(x),
    lambda x: torch.sinc(x),
    lambda x: torch.max(x, torch.zeros_like(x)),
    lambda x: torch.min(x, torch.zeros_like(x)),
    lambda x: torch.sigmoid(x),
    lambda x: torch.log(1 + torch.exp(x)),
    lambda x: torch.exp(-torch.square(x)),
    lambda x: torch.erf(x)
]

autoact_binary_funcs = [
    lambda x, y: x + y,
    lambda x, y: x * y,
    lambda x, y: x - y,
    lambda x, y: x / (y + 1e-7),
    lambda x, y: torch.max(x, y),
    lambda x, y: torch.min(x, y),
    lambda x, y: torch.sigmoid(x) * y,
    #BinaryExpSquare(),
    #BinaryExpAbs(),
    #BinaryAdd()
]

class UnaryFunctionalModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

class BinaryFunctionalModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y):
        return self.fn(x, y)

class AutoActivation(nn.Module):
    """
    """
    def __init__(self, unit_num = 1):
        super().__init__()
        unary1_cand = [UnaryFunctionalModule(fn) for fn in autoact_unary_funcs]
        unary1_cand.extend([UnaryMul(), UnaryAdd()])
        unary2_cand = [UnaryFunctionalModule(fn) for fn in autoact_unary_funcs]
        unary2_cand.extend([UnaryMul(), UnaryAdd()])
        binary_cand = [BinaryFunctionalModule(fn) for fn in autoact_binary_funcs]
        binary_cand.extend([BinaryExpSquare(), BinaryExpAbs(), BinaryAdd()])
        self.unary1 = LayerChoice(unary1_cand, label='one_unary')
        self.unary2 = LayerChoice(unary2_cand, label='one_unary')
        self.binary = LayerChoice(binary_cand)

    def forward(self, x):
        return self.binary(self.unary1(x), self.unary2(x))
