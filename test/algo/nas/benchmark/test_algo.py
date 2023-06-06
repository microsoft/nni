import pytest

from nni.nas.benchmark import *
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.strategy import *

from nni.nas.hub.pytorch import NasBench101, NasBench201

from .prepare import *

# TODO: tune RL and make it work

def test_nasbench101_with_rl():
    pytorch_space = NasBench101()
    benchmark = NasBench101Benchmark()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)

    engine = SequentialExecutionEngine(max_model_count=200)
    strategy = PolicyBasedRL(reward_for_invalid=0)
    strategy(exec_space, engine)
    assert list(strategy.list_models(sort=True, limit=1))[0].metric > 0.94


def test_nasbench201_with_rl():
    pytorch_space = NasBench201()
    benchmark = NasBench201Benchmark()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)

    engine = SequentialExecutionEngine(max_model_count=200)
    strategy = PolicyBasedRL()
    strategy(exec_space, engine)
    assert list(strategy.list_models(sort=True, limit=1))[0].metric > 0.7


@pytest.mark.flaky(reruns=2)
def test_nasbench101_with_evo():
    pytorch_space = NasBench101()
    benchmark = NasBench101Benchmark()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)

    engine = SequentialExecutionEngine(max_model_count=200)
    strategy = RegularizedEvolution(population_size=50, sample_size=25)
    strategy(exec_space, engine)
    assert list(strategy.list_models(sort=True, limit=1))[0].metric > 0.945


def test_nasbench201_with_evo():
    pytorch_space = NasBench201()
    benchmark = NasBench201Benchmark()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)

    engine = SequentialExecutionEngine(max_model_count=200)
    strategy = RegularizedEvolution(population_size=50, sample_size=25)
    strategy(exec_space, engine)
    assert list(strategy.list_models(sort=True, limit=1))[0].metric > 0.73
