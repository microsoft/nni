from nni.mutable import SampleValidationError
from nni.nas.benchmark import *
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.strategy import *

from nni.nas.hub.pytorch import NasBench101


def test_nasbench101():
    pytorch_space = NasBench101()
    benchmark = NasBench101Benchmark()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)

    engine = SequentialExecutionEngine(max_model_count=1000)
    strategy = PolicyBasedRL()
    strategy(exec_space, engine)


test_nasbench101()