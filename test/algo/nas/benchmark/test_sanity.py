from nni.mutable import SampleValidationError
from nni.nas.benchmark import *

from nni.nas.hub.pytorch import NasBench101, NasBench201

from .prepare import *


def test_nasbench101():
    benchmark = NasBench101Benchmark()
    exec_space = BenchmarkModelSpace(benchmark)
    model = exec_space.default()
    with benchmark.mock_runtime(model):
        model.execute()
    assert 0 < model.metric < 1

    good = bad = 0
    for _ in range(30):
        try:
            model = exec_space.random()
            with benchmark.mock_runtime(model):
                model.execute()
            assert 0 < model.metric < 1
            good += 1
        except SampleValidationError:
            bad += 1
    assert good > 0 and bad > 0

    pytorch_space = NasBench101()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)
    model = exec_space.default()
    with benchmark.mock_runtime(model):
        model.execute()
    assert 0 < model.metric < 1


def test_nasbench201():
    benchmark = NasBench201Benchmark()
    exec_space = BenchmarkModelSpace(benchmark)
    model = exec_space.default()
    with benchmark.mock_runtime(model):
        model.execute()
    assert 0 < model.metric < 1

    for _ in range(30):
        model = exec_space.random()
        with benchmark.mock_runtime(model):
            model.execute()
        assert 0 < model.metric < 1

    pytorch_space = NasBench201()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)
    model = exec_space.random()
    with benchmark.mock_runtime(model):
        model.execute()
    assert 0 < model.metric < 1
