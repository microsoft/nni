import sys
import pytest

import nni
from nni.nas.benchmark import download_benchmark

def prepare_benchmark():
    for benchmark in ['nasbench101', 'nasbench201']:
        download_benchmark(benchmark)

@pytest.fixture(autouse=True)
def reset_cached_parameter():
    if sys.platform != 'linux':
        pytest.skip('Benchmark tests are too slow on Windows.')
    nni.trial._params = None
    nni.trial.overwrite_intermediate_seq(0)

if __name__ == '__main__':
    prepare_benchmark()
