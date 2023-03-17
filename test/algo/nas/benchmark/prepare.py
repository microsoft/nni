import sys
import pytest

import nni
from nni.nas.benchmark import download_benchmark

@pytest.fixture(autouse=True, scope='session')
def prepare_benchmark():
    if sys.platform != 'linux':
        pytest.skip('Benchmark tests are too slow on Windows.')
    for benchmark in ['nasbench101', 'nasbench201']:
        download_benchmark(benchmark)

@pytest.fixture(autouse=True)
def reset_cached_parameter():
    nni.trial._params = None
    nni.trial.overwrite_intermediate_seq(0)
