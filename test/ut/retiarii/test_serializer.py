import json
from pathlib import Path
import sys

import torch
from nni.retiarii import json_dumps, json_loads, blackbox
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from imported.model import ImportTest


class Foo:
    def __init__(self, a, b=1):
        self.aa = a
        self.bb = [b + 1 for _ in range(1000)]

    def __eq__(self, other):
        return self.aa == other.aa and self.bb == other.bb


def test_blackbox():
    module = blackbox(Foo, 3)
    assert json_loads(json_dumps(module)) == module
    module = blackbox(Foo, b=2, a=1)
    assert json_loads(json_dumps(module)) == module

    module = blackbox(Foo, Foo(1), 5)
    dumped_module = json_dumps(module)
    assert len(dumped_module) > 200  # should not be too longer if the serialization is correct

    module = blackbox(Foo, blackbox(Foo, 1), 5)
    dumped_module = json_dumps(module)
    assert len(dumped_module) < 200  # should not be too longer if the serialization is correct
    assert json_loads(dumped_module) == module


def test_blackbox_module():
    module = ImportTest(3, 0.5)
    assert json_loads(json_dumps(module)) == module


def test_dataset():
    dataset = blackbox(MNIST, root='data/mnist', train=False, download=True)
    dataloader = blackbox(DataLoader, dataset, batch_size=10)

    dumped_ans = {
        "__type__": "torch.utils.data.dataloader.DataLoader",
        "arguments": {
            "batch_size": 10,
            "dataset": {
                "__type__": "torchvision.datasets.mnist.MNIST",
                "arguments": {"root": "data/mnist", "train": False, "download": True}
            }
        }
    }
    assert json_dumps(dataloader) == json_dumps(dumped_ans)
    dataloader = json_loads(json_dumps(dumped_ans))
    assert isinstance(dataloader, DataLoader)

    dataset = blackbox(MNIST, root='data/mnist', train=False, download=True,
                       transform=blackbox(
                           transforms.Compose,
                           [blackbox(transforms.ToTensor), blackbox(transforms.Normalize, (0.1307,), (0.3081,))]
                       ))
    dataloader = blackbox(DataLoader, dataset, batch_size=10)
    x, y = next(iter(json_loads(json_dumps(dataloader))))
    assert x.size() == torch.Size([10, 1, 28, 28])
    assert y.size() == torch.Size([10])

    dataset = blackbox(MNIST, root='data/mnist', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    dataloader = blackbox(DataLoader, dataset, batch_size=10)
    x, y = next(iter(json_loads(json_dumps(dataloader))))
    assert x.size() == torch.Size([10, 1, 28, 28])
    assert y.size() == torch.Size([10])


def test_type():
    assert json_dumps(torch.optim.Adam) == '{"__typename__": "torch.optim.adam.Adam"}'
    assert json_loads('{"__typename__": "torch.optim.adam.Adam"}') == torch.optim.Adam
    assert json_dumps(Foo) == '{"__typename__": "test_serializer.Foo"}'


if __name__ == '__main__':
    test_blackbox()
    test_blackbox_module()
    test_dataset()
    test_type()
