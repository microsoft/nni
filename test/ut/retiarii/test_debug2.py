import multiprocessing
import time

import nni
from torchvision.datasets import MNIST


class MyMNIST(MNIST):

    def _load_data(self):
        print('loading data', flush=True)
        res = super()._load_data()
        print('load data complete', flush=True)
        return res


def test_main_process():
    MyMNIST('data/mnist', train=True, download=True, transform=None)


def _test_dry_run():
    print('dry run', flush=True)
    MyMNIST('data/mnist', train=True, download=True, transform=None)
    print('dry run complete', flush=True)


def test_exp_exit_without_stop():
    process = multiprocessing.Process(
        target=_test_dry_run,
    )
    process.start()
    print('Waiting for first dry run in sub-process.')
    for _ in range(30):
        if process.is_alive():
            time.sleep(1)
        else:
            assert process.exitcode == 0
            return

    raise ValueError()


print('no trace', flush=True)
test_main_process()
test_exp_exit_without_stop()
test_exp_exit_without_stop()
