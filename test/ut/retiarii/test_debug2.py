import multiprocessing
import time

import nni
from torchvision.datasets import MNIST


def test_main_process():
    MNIST('data/mnist', train=True, download=True, transform=None)


def _test_dry_run():
    print('dry run')
    MNIST('data/mnist', train=True, download=True, transform=None)
    print('dry run complete')


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


test_main_process()
test_exp_exit_without_stop()
