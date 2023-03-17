import pytest

from nni.nas.benchmark import download_benchmark

@pytest.fixture(autouse=True, scope='session')
def prepare_benchmark():
    for benchmark in ['nasbench101', 'nasbench201']:
        download_benchmark(benchmark)
