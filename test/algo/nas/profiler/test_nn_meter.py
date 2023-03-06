import pytest

import torch
from torch import nn

import nni
from nni.mutable import Categorical
from nni.nas.hub.pytorch import *
from nni.nas.hub.pytorch.mobilenetv3 import SqueezeExcite
from nni.nas.hub.pytorch.proxylessnas import ConvBNReLU, DepthwiseSeparableConv, InvertedResidual
from nni.nas.nn.pytorch import MutableConv2d, MutableLinear, LayerChoice, Repeat
from nni.nas.profiler.pytorch.nn_meter import NnMeterProfiler, to_onnx, combinations
from nni.nas.profiler.pytorch.utils import MutableShape
from nn_meter import load_latency_predictor

from ut.nas.nn.models import MODELS


@pytest.fixture(scope='module')
def predictor():
    return load_latency_predictor('cortexA76cpu_tflite21')


def test_combinations():
    net = LayerChoice([nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=True)], label='x')
    shape = (MutableShape(3), MutableShape(2, nni.choice('y', [3, 4])), MutableShape(nni.choice('x', [0, 1]) + 1))

    counter = 0
    bias_true = 0
    shape_four = 0
    for condition, n, inp in combinations(net, shape):
        if counter == 0:
            assert condition.evaluate({'x': 0, 'y': 3})
            assert not condition.evaluate({'x': 1, 'y': 3})
        if counter == 3:
            assert condition.evaluate({'x': 1, 'y': 4})
            assert not condition.evaluate({'x': 1, 'y': 3})
        bias_true += (n.bias is not None)

        assert len(inp) == 3
        if inp[1].size(1) == 4:
            shape_four += 1
        assert inp[2].size(0) == int(n.bias is not None) + 1
        counter += 1
    assert counter == 4
    assert bias_true == 2
    assert shape_four == 2

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = Repeat(
                lambda d: LayerChoice([
                    nn.Linear(3, 3, bias=False),
                    nn.Linear(3, 3, bias=True)
                ],
                label=f'd{d}'
            ), (1, 2), label='rep')

    net = Net()
    shape = MutableShape(2, nni.choice('rep', [1, 2]))
    assert len(list(combinations(net, shape))) == 8

    assert len(list(combinations(nni.choice('d1', [0, 1]), shape))) == 4
    assert len(list(combinations(nni.choice('rep', [1, 2]), shape))) == 2


def test_linear(predictor):
    net = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    profiler = NnMeterProfiler(net, torch.randn(1, 3), predictor)
    assert profiler.profile({}) < 1

    net = MutableLinear(1000, Categorical([2, 3000], label='x'))
    profiler = NnMeterProfiler(net, torch.randn(1, 1000), predictor)
    assert profiler.profile({'x': 2}) < profiler.profile({'x': 3000})


def test_conv2d(predictor):
    net = nn.Conv2d(3, 4, 1)
    profiler = NnMeterProfiler(net, torch.randn(1, 3, 3, 3), predictor)
    assert profiler.profile({}) < 1

    net = nn.Conv2d(100, 200, 3)
    profiler = NnMeterProfiler(net, torch.randn(1, 100, 100, 100), predictor)
    assert profiler.profile({}) > 1


def test_layerchoice(predictor):
    net = LayerChoice([
        nn.Sequential(nn.Conv2d(32, 128, 1), nn.Conv2d(128, 32, 1)),
        nn.Sequential(nn.Conv2d(32, 1, 1), nn.Conv2d(1, 32, 1)),
    ], label='x')
    profiler = NnMeterProfiler(net, torch.randn(1, 32, 32, 32), predictor)
    assert profiler.profile({'x': 0}) > 2 * profiler.profile({'x': 1})


def test_layerchoice_nested(predictor):
    net = LayerChoice([
        nn.Sequential(MutableConv2d(32, nni.choice('a', [128, 256]), 1), MutableConv2d(nni.choice('a', [128, 256]), 32, 1)),
        nn.Sequential(MutableConv2d(32, nni.choice('b', [1, 32]), 1), MutableConv2d(nni.choice('b', [1, 32]), 32, 1)),
    ], label='y')
    profiler = NnMeterProfiler(net, torch.randn(1, 32, 32, 32), predictor)
    result = profiler.profile({'y': 0, 'a': 128, 'b': 1})
    assert 0 < profiler.profile({'y': 0, 'a': 128, 'b': 1}) < profiler.profile({'y': 0, 'a': 256, 'b': 1})
    assert profiler.profile({'y': 1, 'a': 128, 'b': 1}) == profiler.profile({'y': 1, 'a': 256, 'b': 1})
    assert profiler.profile({'y': 0, 'a': 128, 'b': 32}) > profiler.profile({'y': 1, 'a': 256, 'b': 1})

    class CustomProfiler(NnMeterProfiler):
        def estimate_layerchoice_latency(self, *args):
            raise RuntimeError()

        def is_leaf_module(self, module: nn.Module) -> bool:
            return super().is_leaf_module(module) or isinstance(module, LayerChoice)

    profiler = CustomProfiler(net, torch.randn(1, 32, 32, 32), predictor)
    assert result * 0.95 < profiler.profile({'y': 0, 'a': 128, 'b': 1}) < result * 1.05


def test_repeat_nested(predictor):
    net = Repeat(
        lambda d: LayerChoice([
            nn.Sequential(nn.Conv2d(32, 128, 1), nn.Conv2d(128, 32, 1)),
            nn.Sequential(nn.Conv2d(32, 1, 1), nn.Conv2d(1, 32, 1)),
        ], label=f'd{d}'), (1, 3), label='rep' 
    )

    profiler = NnMeterProfiler(net, torch.randn(1, 32, 32, 32), predictor)
    assert profiler.profile({'rep': 1, 'd0': 0, 'd1': 0, 'd2': 0}) < profiler.profile({'rep': 3, 'd0': 0, 'd1': 0, 'd2': 0})
    assert profiler.profile({'rep': 3, 'd0': 0, 'd1': 1, 'd2': 0}) < profiler.profile({'rep': 3, 'd0': 0, 'd1': 0, 'd2': 0})

    result = profiler.profile({'rep': 2, 'd0': 0, 'd1': 1, 'd2': 0})

    class CustomProfiler(NnMeterProfiler):
        def estimate_repeat_latency(self, *args):
            raise RuntimeError()

        def estimate_layerchoice_latency(self, *args):
            raise RuntimeError()

        def is_leaf_module(self, module: nn.Module) -> bool:
            return super().is_leaf_module(module) or isinstance(module, Repeat)

    profiler = CustomProfiler(net, torch.randn(1, 32, 32, 32), predictor)
    # NOTE: I'm not sure. Not true if conv-conv fusion?
    assert result * 0.95 < profiler.profile({'rep': 2, 'd0': 0, 'd1': 1, 'd2': 0}) < result * 1.05


@pytest.mark.parametrize('model', [m for m in MODELS if not m.startswith('cell')])
def test_e2e_simple(model, predictor):
    model_space = MODELS[model]()
    if model == 'custom_op':
        with pytest.raises(RuntimeError, match='Shape inference failed'):
            NnMeterProfiler(model_space, torch.randn(1, 1, 28, 28), predictor)
        return
    if model == 'multihead_attention':
        pytest.skip('MultiheadAttention is not supported by nn-meter yet.')
    else:
        profiler = NnMeterProfiler(model_space, torch.randn(1, 1, 28, 28), predictor)
    sample = {}
    for _ in model_space.grid(memo=sample):
        # TODO: Check the result
        print(sample, profiler.profile(sample))


def test_proxylessnas(predictor):
    custom_leaf_types = (ConvBNReLU, DepthwiseSeparableConv, InvertedResidual)

    model_space = ProxylessNAS()
    sample = {}
    model = model_space.random(sample)

    profiler = NnMeterProfiler(
        model_space, torch.randn(1, 3, 224, 224), predictor,
        custom_leaf_types=custom_leaf_types
    )
    result1 = profiler.profile(sample)

    profiler = NnMeterProfiler(
        model, torch.randn(1, 3, 224, 224), predictor,
        custom_leaf_types=custom_leaf_types
    )
    result2 = profiler.profile({})

    assert result1 > 0
    assert abs(result1 - result2) < 1


def test_mobilenetv3(predictor):
    custom_leaf_types = (ConvBNReLU, DepthwiseSeparableConv, InvertedResidual, SqueezeExcite)

    model_space = MobileNetV3Space(width_multipliers=(0.5, 1.0), expand_ratios=(3., 6.))
    sample = {}
    model = model_space.random(sample)

    profiler = NnMeterProfiler(
        model_space, torch.randn(1, 3, 224, 224), predictor,
        custom_leaf_types=custom_leaf_types,
        simplify_shapes=True
    )
    result1 = profiler.profile(sample)

    profiler = NnMeterProfiler(
        model, torch.randn(1, 3, 224, 224), predictor,
        custom_leaf_types=custom_leaf_types,
        simplify_shapes=True
    )
    result2 = profiler.profile({})

    assert result1 > 0
    assert abs(result1 - result2) < 1
