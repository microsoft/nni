import math
from pathlib import Path
import re
import sys

import nni
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from nni.common.serializer import is_traceable

if True:  # prevent auto formatting
    sys.path.insert(0, Path(__file__).parent.as_posix())
    from imported.model import ImportTest


@nni.trace
class SimpleClass:
    def __init__(self, a, b=1):
        self._a = a
        self._b = b


class UnserializableSimpleClass:
    def __init__(self):
        self._a = 1


def test_simple_class():
    instance = SimpleClass(1, 2)
    assert instance._a == 1
    assert instance._b == 2

    dump_str = nni.dump(instance)
    assert '"__kwargs__": {"a": 1, "b": 2}' in dump_str
    assert '"__symbol__"' in dump_str
    instance = nni.load(dump_str)
    assert instance._a == 1
    assert instance._b == 2


def test_external_class():
    from collections import OrderedDict
    d = nni.trace(kw_only=False)(OrderedDict)([('a', 1), ('b', 2)])
    assert d['a'] == 1
    assert d['b'] == 2
    dump_str = nni.dump(d)
    assert dump_str == '{"a": 1, "b": 2}'

    conv = nni.trace(torch.nn.Conv2d)(3, 16, 3)
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3)
    assert nni.dump(conv) == \
        r'{"__symbol__": "path:torch.nn.modules.conv.Conv2d", ' \
        r'"__kwargs__": {"in_channels": 3, "out_channels": 16, "kernel_size": 3}}'

    conv = nni.load(nni.dump(conv))
    assert conv.kernel_size == (3, 3)


def test_nested_class():
    a = SimpleClass(1, 2)
    b = SimpleClass(a)
    assert b._a._a == 1
    dump_str = nni.dump(b)
    b = nni.load(dump_str)
    assert 'SimpleClass object at' in repr(b)
    assert b._a._a == 1


def test_unserializable():
    a = UnserializableSimpleClass()
    dump_str = nni.dump(a)
    a = nni.load(dump_str)
    assert a._a == 1


def test_function():
    t = nni.trace(math.sqrt, kw_only=False)(3)
    assert 1 < t < 2
    assert t.trace_symbol == math.sqrt
    assert t.trace_args == [3]
    print(nni.dump(t))
    t = nni.load(nni.dump(t))
    assert 1 < t < 2
    assert is_traceable(t)

    def simple_class_factory(bb=3.):
        return SimpleClass(1, bb)

    t = nni.trace(simple_class_factory)(4)


class Foo:
    def __init__(self, a, b=1):
        self.aa = a
        self.bb = [b + 1 for _ in range(1000)]

    def __eq__(self, other):
        return self.aa == other.aa and self.bb == other.bb


def test_custom_class():
    module = nni.trace(Foo)(3)
    assert nni.load(nni.dump(module)) == module
    module = nni.trace(Foo)(b=2, a=1)
    assert nni.load(nni.dump(module)) == module

    module = nni.trace(Foo)(Foo(1), 5)
    dumped_module = nni.dump(module)
    assert len(dumped_module) > 200  # should not be too longer if the serialization is correct

    module = nni.trace(Foo)(nni.trace(Foo)(1), 5)
    dumped_module = nni.dump(module)
    assert len(dumped_module) < 200  # should not be too longer if the serialization is correct
    assert nni.load(dumped_module) == module


class Foo:
    def __init__(self, a, b=1):
        self.aa = a
        self.bb = [b + 1 for _ in range(1000)]

    def __eq__(self, other):
        return self.aa == other.aa and self.bb == other.bb


def test_basic_unit():
    module = ImportTest(3, 0.5)
    assert nni.load(nni.dump(module)) == module

    import nni.retiarii.nn.pytorch as nn
    module = nn.Conv2d(3, 10, 3, bias=False)
    assert nni.load(nni.dump(module)).bias is None


def test_dataset():
    dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True)
    dataloader = nni.trace(DataLoader)(dataset, batch_size=10)

    dumped_ans = {
        "__symbol__": "torch.utils.data.dataloader.DataLoader",
        "__kwargs__": {
            "batch_size": 10,
            "dataset": {
                "__symbol__": "torchvision.datasets.mnist.MNIST",
                "__kwargs__": {"root": "data/mnist", "train": False, "download": True}
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
    test_simple_class()
    test_external_class()
    test_nested_class()
    test_unserializable()
    test_basic_unit()
    test_function()
