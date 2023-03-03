import pytest

import torch
from torch import nn

import nni
import nni.nas.nn.pytorch as nas_nn
from nni.mutable import Categorical
from nni.nas.profiler.pytorch.flops import FlopsParamsProfiler, FlopsProfiler, NumParamsProfiler, FlopsResult, register_flops_formula

from ut.nas.nn.models import MODELS


def test_conv():
    net = nn.Sequential(nn.Conv2d(3, 2, 3, bias=True))
    profiler = FlopsParamsProfiler(net, torch.randn(3, 224, 224))
    assert profiler.profile({}).flops == 2759904

    net = nas_nn.MutableConv2d(1, 1, Categorical([3, 5], label='a'), bias=False)
    profiler = FlopsParamsProfiler(net, torch.randn(1, 1, 5, 5))
    assert profiler.profile({'a': 3}).flops == 81
    assert profiler.profile({'a': 5}).flops == 25
    assert profiler.profile({'a': 3}).params == 9

    net = nas_nn.MutableConv2d(1, 1, Categorical([3, 5], label='a'), bias=True)
    profiler = FlopsParamsProfiler(net, torch.randn(1, 1, 5, 5), count_bias=False)
    assert profiler.profile({'a': 3}).flops == 81
    assert profiler.profile({'a': 5}).flops == 25
    assert profiler.profile({'a': 3}).params == 10


def test_fc():
    net = nn.Sequential(nn.Linear(3, 2, bias=True))
    profiler = FlopsParamsProfiler(net, torch.randn(3))
    assert profiler.profile({}).flops == 8

    profiler = FlopsProfiler(net, torch.randn(3))
    assert profiler.profile({}) == 8

    profiler = NumParamsProfiler(net, torch.randn(3))
    assert profiler.profile({}) == 8

    profiler = FlopsProfiler(net, torch.randn(3), count_bias=False)
    assert profiler.profile({}) == 6

    # Ignore batch dim
    profiler = FlopsProfiler(net, torch.randn(2, 3))
    assert profiler.profile({}) == 8

    profiler = FlopsProfiler(net, torch.randn(2, 5, 3))
    assert profiler.profile({}) == 40

    net = nas_nn.MutableLinear(3, nni.choice('x', [1, 3, 5]), bias=False)
    profiler = FlopsProfiler(net, torch.randn(3))
    assert profiler.profile({'x': 1}) == 3
    assert profiler.profile({'x': 5}) == 15


def test_bn_relu():
    net = nn.Sequential(nn.BatchNorm2d(3, affine=True), nn.ReLU())

    profiler = FlopsParamsProfiler(net, torch.randn(2, 3, 12, 12))
    assert profiler.profile({}).flops == 3 * 144 * (4 + 1)
    assert profiler.profile({}).params == 6

    profiler = FlopsParamsProfiler(net, torch.randn(1, 3, 12, 12), count_normalization=False)
    assert profiler.profile({}).flops == 3 * 144 * 1
    assert profiler.profile({}).params == 0

    profiler = FlopsParamsProfiler(net, torch.randn(3, 3, 12, 12), count_activation=False)
    assert profiler.profile({}).flops == 3 * 144 * 4


def test_mhattn():
    mhattn = nn.MultiheadAttention(6, 3)
    query = torch.randn(8, 1, 6)
    key = torch.randn(7, 1, 6)
    value = torch.randn(7, 1, 6)
    profiler = FlopsParamsProfiler(mhattn, (query, key, value))
    result = profiler.profile({})
    assert result.flops == 2148
    assert result.params == sum(p.numel() for p in mhattn.parameters())

    mhattn = nn.MultiheadAttention(6, 3, batch_first=True, kdim=5, vdim=4, bias=True)
    query = torch.randn(1, 8, 6)
    key = torch.randn(1, 7, 5)
    value = torch.randn(1, 7, 4)
    profiler = FlopsParamsProfiler(mhattn, (query, key, value))
    result = profiler.profile({})
    assert result.flops == 2022
    assert result.params == 147  # torch created extra 3 parameters. Cannot check.

    mhattn = nn.MultiheadAttention(6, 3, kdim=5, vdim=4, bias=False)
    query = torch.randn(8, 1, 6)
    key = torch.randn(7, 1, 5)
    value = torch.randn(7, 1, 4)
    profiler = FlopsParamsProfiler(mhattn, (query, key, value))
    result = profiler.profile({})
    assert result.flops == 1842
    assert result.params == sum(p.numel() for p in mhattn.parameters())


def test_layerchoice():
    net = nas_nn.LayerChoice([
        nn.Linear(3, 2, bias=False),
        nn.Linear(3, 3, bias=False)
    ], label='a')
    profiler = FlopsParamsProfiler(net, torch.randn(3))
    assert profiler.profile({'a': 0}).flops == 6
    assert profiler.profile({'a': 1}).flops == 9


def test_inputchoice():
    net = nas_nn.InputChoice(3, 1, label='a')
    profiler = FlopsParamsProfiler(net, [torch.randn(3), torch.randn(3)])
    assert profiler.profile({'a': [1]}).flops == 0


def test_repeat():
    net = nas_nn.Repeat(nn.Linear(3, 3, bias=False), nni.choice('rep', [0, 1, 2, 4]))
    profiler = FlopsParamsProfiler(net, torch.randn(1, 3))
    assert profiler.profile({'rep': 0}).flops == 0
    assert profiler.profile({'rep': 1}).flops == 9
    assert profiler.profile({'rep': 2}).flops == 18
    assert profiler.profile({'rep': 4}).flops == 36
    assert profiler.profile({'rep': 2}).params == 18


def test_custom_formula():
    class Anything(nn.Module):
        def forward(self, x):
            return x

    def custom_formula(module, inputs, outputs):
        return FlopsResult(1., 2.)

    net = Anything()
    register_flops_formula(Anything, custom_formula)
    profiler = FlopsParamsProfiler(net, torch.randn(1, 3))
    assert profiler.profile({}) == FlopsResult(1., 2.)

    def new_custom_formula(module, inputs, outputs):
        return FlopsResult(2., 3.)

    Anything._count_flops = new_custom_formula
    profiler = FlopsParamsProfiler(net, torch.randn(1, 3))
    assert profiler.profile({}) == FlopsResult(2., 3.)


@pytest.mark.parametrize('model', [m for m in MODELS if not m.startswith('cell')])
def test_e2e_simple(model):
    model_space = MODELS[model]()
    if model == 'custom_op':
        with pytest.raises(RuntimeError, match='Shape inference failed'):
            FlopsParamsProfiler(model_space, torch.randn(1, 1, 28, 28))
        return
    if model == 'multihead_attention':
        profiler = FlopsParamsProfiler(model_space, (
            (torch.randn(2, 128), torch.randn(2, 128), torch.randn(2, 128)),
        ))
    else:
        profiler = FlopsParamsProfiler(model_space, torch.randn(1, 1, 28, 28))
    sample = {}
    for _ in model_space.grid(memo=sample):
        # TODO: Check the result
        print(sample, profiler.flops_result.freeze(sample))
