import multiprocessing
import time
import inspect

import torch
import numpy as np

import nni
from torchvision.datasets import MNIST

import os
from torchvision.datasets.mnist import read_image_file, read_label_file, read_sn3_pascalvincent_tensor, get_int, SN3_PASCALVINCENT_TYPEMAP


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
        print(inspect.getsource(read_sn3_pascalvincent_tensor), flush=True)
        with open(os.path.join(self.raw_folder, label_file), 'rb') as f:
            print(len(f.read()), flush=True)
        print('hello', flush=True)

        path = os.path.join(self.raw_folder, label_file)

        print('hello', 1, flush=True)
        with open(path, "rb") as f:
            data = f.read()
        # parse
        print('hello', 2, flush=True)
        magic = get_int(data[0:4])
        nd = magic % 256
        ty = magic // 256
        print('hello', 3, flush=True)
        assert 1 <= nd <= 3
        assert 8 <= ty <= 14
        print('hello', 4, flush=True)
        m = SN3_PASCALVINCENT_TYPEMAP[ty]
        print('hello', 5, flush=True)
        s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
        print('hello', 6, flush=True)
        parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
        print('hello', 7, flush=True)
        parsed = torch.from_numpy(parsed.astype(m[2])).view(*s)

        print('hello', 'again', flush=True)
        read_sn3_pascalvincent_tensor(os.path.join(self.raw_folder, label_file), strict=False)

        print('read_sn3_pascalvincent_tensor complete', flush=True)
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
