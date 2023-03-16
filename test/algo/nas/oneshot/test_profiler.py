import pytest

import torch
from torch.nn import Conv2d

from nni.nas.nn import ModelSpace, LayerChoice
from nni.nas.oneshot.pytorch.profiler import RangeProfilerFilter, ExpectationProfilerPenalty, SampleProfilerPenalty
from nni.nas.profiler.pytorch.flops import FlopsProfiler


class Net(ModelSpace):
    def __init__(self):
        super().__init__()
        self.conv = LayerChoice([
            Conv2d(1, 10, 3, padding=1, bias=False),
            Conv2d(1, 10, 5, padding=2, bias=False),
            Conv2d(1, 10, 7, padding=3, bias=False)
        ], label='conv')

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def profiler():
    net = Net()
    profiler = FlopsProfiler(net, torch.randn(1, 1, 10, 10))
    # (9k, 25k, 49k)
    return profiler


def test_range_filter(profiler):
    filter = RangeProfilerFilter(profiler, min=10000)
    assert filter({'conv': 1})
    assert filter({'conv': 2})
    assert not filter({'conv': 0})

    filter = RangeProfilerFilter(profiler, max=30000)
    assert filter({'conv': 0})
    assert filter({'conv': 1})
    assert not filter({'conv': 2})

    filter = RangeProfilerFilter(profiler, min=10000, max=30000)
    assert not filter({'conv': 0})
    assert filter({'conv': 1})
    assert not filter({'conv': 2})

    with pytest.raises(ValueError, match='both None'):
        RangeProfilerFilter(profiler, min=None, max=None)


def test_expectation_penalty(profiler):
    penalty = ExpectationProfilerPenalty(profiler, 20000, 2.)
    loss, details = penalty(42., {'conv': {0: 0.2, 1: 0.3, 2: 0.5}})
    assert details['loss_original'] == 42.
    assert details['penalty'] == 33800
    assert details['normalized_penalty'] == 0.69
    assert loss == 42. + 0.69 * 2.

    prob = torch.tensor([0.2, 0.3, 0.5])
    loss, details = penalty(torch.tensor(42.), {'conv': {i: prob[i] for i in range(3)}})
    assert abs(loss.item() - (42. + 0.69 * 2.)) < 1e-4

    penalty = ExpectationProfilerPenalty(profiler, 40000, -2., nonlinear='negative')
    loss, details = penalty(42., {'conv': {0: 0.2, 1: 0.3, 2: 0.5}})
    assert details['normalized_penalty'] == 33800 / 40000 - 1
    assert loss == 42. + 0.155 * 2.

    loss, details = penalty(42., {'conv': {0: 0., 1: 0., 2: 1.}})
    assert details['normalized_penalty'] == 0
    assert loss == 42.

    penalty = ExpectationProfilerPenalty(profiler, 40000, 2., nonlinear='absolute', aggregate='mul')
    loss, details = penalty(42., {'conv': {0: 0.2, 1: 0.3, 2: 0.5}})
    assert details['normalized_penalty'] == abs(33800 / 40000 - 1)
    assert loss == 42. * (1 + 0.155) ** 2
    
    penalty = ExpectationProfilerPenalty(profiler, 30000, 2., nonlinear='positive', aggregate='mul')
    loss, details = penalty(42., {'conv': {0: 0.2, 1: 0.3, 2: 0.5}})
    assert details['normalized_penalty'] == 33800 / 30000 - 1
    assert loss == 42. * (33800 / 30000) ** 2

    loss, details = penalty(42., {'conv': {0: 1., 1: 0., 2: 0.}})
    assert details['normalized_penalty'] == 0
    assert loss == 42.


def test_sample_penalty(profiler):
    penalty = SampleProfilerPenalty(profiler, 20000, 2.)
    loss, details = penalty(42., {'conv': 1})
    assert details['loss_original'] == 42.
    assert details['penalty'] == 25000
    assert details['normalized_penalty'] == 0.25
    assert loss == 42. + 0.25 * 2.
