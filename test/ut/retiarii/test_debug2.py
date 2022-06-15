import multiprocessing
import time

import nni
from torchvision.datasets import MNIST

import os
from torchvision.datasets.mnist import read_image_file, read_label_file


class MyMNIST(MNIST):

    def _load_data(self):
        print('loading data', flush=True)
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        assert os.path.exists(os.path.join(self.raw_folder, image_file))
        data = read_image_file(os.path.join(self.raw_folder, image_file))
        print('loading data complete', flush=True)

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        assert os.path.exists(os.path.join(self.raw_folder, label_file))
        print('target exist', flush=True)
        with open(os.path.join(self.raw_folder, label_file), 'rb') as f:
            print(len(f.read()), flush=True)
        targets = read_label_file(os.path.join(self.raw_folder, label_file))
        print('load targets complete', flush=True)
        return data, targets


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
