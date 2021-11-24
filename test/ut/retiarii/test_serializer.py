from imported.model import ImportTest
import math
from pathlib import Path
import re
import sys

import nni
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

sys.path.insert(0, Path(__file__).parent.as_posix())


class Foo:
    def __init__(self, a, b=1):
        self.aa = a
        self.bb = [b + 1 for _ in range(1000)]

    def __eq__(self, other):
        return self.aa == other.aa and self.bb == other.bb


def test_basic_unit():
    module = ImportTest(3, 0.5)
    assert nni.load(nni.dump(module)) == module


def test_dataset():
    dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True)
    dataloader = nni.trace(DataLoader)(dataset, batch_size=10)

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
    assert nni.dump(dataloader) == nni.dump(dumped_ans)
    dataloader = nni.load(nni.dump(dumped_ans))
    assert isinstance(dataloader, DataLoader)

    dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True,
                               transform=nni.trace(transforms.Compose)(
                                   [nni.trace(transforms.ToTensor), nni.trace(transforms.Normalize, (0.1307,), (0.3081,))]
                               ))
    dataloader = nni.trace(DataLoader)(dataset, batch_size=10)
    x, y = next(iter(nni.load(nni.dump(dataloader))))
    assert x.size() == torch.Size([10, 1, 28, 28])
    assert y.size() == torch.Size([10])

    dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True,
                               transform=nni.trace(transforms.Compose)(
                                   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                               ))
    dataloader = nni.trace(DataLoader)(dataset, batch_size=10)
    x, y = next(iter(nni.load(nni.dump(dataloader))))
    assert x.size() == torch.Size([10, 1, 28, 28])
    assert y.size() == torch.Size([10])


def test_type():
    assert nni.dump(torch.optim.Adam) == '{"__nni_type__": "path:torch.optim.adam.Adam"}'
    assert nni.load('{"__nni_type__": "path:torch.optim.adam.Adam"}') == torch.optim.Adam
    assert re.match(r'{"__nni_type__": "bytes:(.*)"}', nni.dump(Foo))
    assert nni.dump(math.floor) == '{"__nni_type__": "path:math.floor"}'
    assert nni.load('{"__nni_type__": "path:math.floor"}') == math.floor


def test_lightning_earlystop():
    import nni.retiarii.evaluator.pytorch.lightning as pl
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    trainer = pl.Trainer(callbacks=[nni.trace(EarlyStopping)(monitor="val_loss")])
    trainer = nni.load(nni.dump(trainer))
    print(trainer.get())  # FIXME AttributeError: 'SerializableObject' object has no attribute 'on_init_start'


if __name__ == '__main__':
    # test_basic_unit()
    # test_dataset()
    # test_type()
    test_lightning_earlystop()
