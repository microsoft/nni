from collections import OrderedDict
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import nni
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from nni.common.serializer import is_traceable

if True:  # prevent auto formatting
    sys.path.insert(0, Path(__file__).parent.as_posix())
    # this test cannot be directly put in this file. It will cause syntax error for python <= 3.7.
    if tuple(sys.version_info) >= (3, 8):
        from imported._test_serializer_py38 import test_positional_only


def test_ordered_json():
    items = [
        ('a', 1),
        ('c', 3),
        ('b', 2),
    ]
    orig = OrderedDict(items)
    json = nni.dump(orig)
    loaded = nni.load(json)
    assert list(loaded.items()) == items


@nni.trace
class SimpleClass:
    def __init__(self, a, b=1):
        self._a = a
        self._b = b


@nni.trace
class EmptyClass:
    pass


class CustomizeLoadDump:
    def __init__(self, a, b=1):
        self._a = a
        self._b = b

    def _dump(self):
        return {
            'a': self._a,
            'any': self._b
        }

    @staticmethod
    def _load(*, a, any):
        return CustomizeLoadDump(a, any)


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


def test_customize_class():
    instance = CustomizeLoadDump(1, 2)
    dump_str = nni.dump(instance)
    # FIXME


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
    t = nni.load(nni.dump(t))
    assert 1 < t < 2
    assert not is_traceable(t)  # trace not recovered, expected, limitation

    def simple_class_factory(bb=3.):
        return SimpleClass(1, bb)

    t = nni.trace(simple_class_factory)(4)
    ts = nni.dump(t)
    assert '__kwargs__' in ts
    t = nni.load(ts)
    assert t._a == 1
    assert is_traceable(t)
    t = t.trace_copy()
    assert is_traceable(t)
    assert t.trace_symbol(10)._b == 10
    assert t.trace_kwargs['bb'] == 4
    assert is_traceable(t.trace_copy())


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
    module = nni.load(dumped_module)
    assert module.bb[0] == module.bb[999] == 6

    module = nni.trace(Foo)(nni.trace(Foo)(1), 5)
    dumped_module = nni.dump(module)
    assert nni.load(dumped_module) == module


class Foo:
    def __init__(self, a, b=1):
        self.aa = a
        self.bb = [b + 1 for _ in range(1000)]

    def __eq__(self, other):
        return self.aa == other.aa and self.bb == other.bb


def test_dataset():
    dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True)
    dataloader = nni.trace(DataLoader)(dataset, batch_size=10)

    dumped_ans = {
        "__symbol__": "path:torch.utils.data.dataloader.DataLoader",
        "__kwargs__": {
            "dataset": {
                "__symbol__": "path:torchvision.datasets.mnist.MNIST",
                "__kwargs__": {"root": "data/mnist", "train": False, "download": True}
            },
            "batch_size": 10
        }
    }
    print(nni.dump(dataloader))
    print(nni.dump(dumped_ans))
    assert nni.dump(dataloader) == nni.dump(dumped_ans)
    dataloader = nni.load(nni.dump(dumped_ans))
    assert isinstance(dataloader, DataLoader)

    dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True,
                               transform=nni.trace(transforms.Compose)([
                                   nni.trace(transforms.ToTensor)(),
                                   nni.trace(transforms.Normalize)((0.1307,), (0.3081,))
                               ]))
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


def test_pickle():
    pickle.dumps(EmptyClass())
    obj = SimpleClass(1)
    obj = pickle.loads(pickle.dumps(obj))

    assert obj._a == 1
    assert obj._b == 1

    obj = SimpleClass(1)
    obj.xxx = 3
    obj = pickle.loads(pickle.dumps(obj))
    assert obj.xxx == 3


@pytest.mark.skipif(sys.platform != 'linux', reason='https://github.com/microsoft/nni/issues/4434')
def test_multiprocessing_dataloader():
    # check whether multi-processing works
    # it's possible to have pickle errors
    dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True,
                               transform=nni.trace(transforms.Compose)(
                                   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                               ))
    import nni.nas.evaluator.pytorch.lightning as pl
    dataloader = pl.DataLoader(dataset, batch_size=10, num_workers=2)
    x, y = next(iter(dataloader))
    assert x.size() == torch.Size([10, 1, 28, 28])
    assert y.size() == torch.Size([10])


def _test_multiprocessing_dataset_worker(dataset):
    if sys.platform == 'linux':
        # on non-linux, the loaded object will become non-traceable
        # due to an implementation limitation
        assert is_traceable(dataset)
    else:
        from torch.utils.data import Dataset
        assert isinstance(dataset, Dataset)


def test_multiprocessing_dataset():
    from torch.utils.data import Dataset

    dataset = nni.trace(Dataset)()

    import multiprocessing
    process = multiprocessing.Process(target=_test_multiprocessing_dataset_worker, args=(dataset, ))
    process.start()
    process.join()
    assert process.exitcode == 0


def test_type():
    assert nni.dump(torch.optim.Adam) == '{"__nni_type__": "path:torch.optim.adam.Adam"}'
    assert nni.load('{"__nni_type__": "path:torch.optim.adam.Adam"}') == torch.optim.Adam
    assert Foo == nni.load(nni.dump(Foo))
    assert nni.dump(math.floor) == '{"__nni_type__": "path:math.floor"}'
    assert nni.load('{"__nni_type__": "path:math.floor"}') == math.floor


def test_lightning_earlystop():
    import nni.nas.evaluator.pytorch.lightning as pl
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    trainer = pl.Trainer(callbacks=[nni.trace(EarlyStopping)(monitor="val_loss")])
    pickle_size_limit = 4096 if sys.platform == 'linux' else 32768
    trainer = nni.load(nni.dump(trainer, pickle_size_limit=pickle_size_limit))
    assert any(isinstance(callback, EarlyStopping) for callback in trainer.callbacks)


def test_pickle_trainer():
    import nni.nas.evaluator.pytorch.lightning as pl
    from pytorch_lightning import Trainer
    trainer = pl.Trainer(max_epochs=1)
    data = pickle.dumps(trainer)
    trainer = pickle.loads(data)
    assert isinstance(trainer, Trainer)


def test_generator():
    import torch.nn as nn
    import torch.optim as optim

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 10, 1)

        def forward(self, x):
            return self.conv(x)

    model = Net()
    optimizer = nni.trace(optim.Adam)(model.parameters())
    print(optimizer.trace_kwargs)


def test_arguments_kind():
    def foo(a, b, *c, **d):
        pass

    d = nni.trace(foo)(1, 2, 3, 4)
    assert d.trace_args == [1, 2, 3, 4]
    assert d.trace_kwargs == {}

    d = nni.trace(foo)(a=1, b=2)
    assert d.trace_kwargs == dict(a=1, b=2)

    d = nni.trace(foo)(1, b=2)
    # this is not perfect, but it's safe
    assert d.trace_kwargs == dict(a=1, b=2)

    def foo(a, *, b=3, c=5):
        pass

    d = nni.trace(foo)(1, b=2, c=3)
    assert d.trace_kwargs == dict(a=1, b=2, c=3)

    import torch.nn as nn
    lstm = nni.trace(nn.LSTM)(2, 2)
    assert lstm.input_size == 2
    assert lstm.hidden_size == 2
    assert lstm.trace_args == [2, 2]

    lstm = nni.trace(nn.LSTM)(input_size=2, hidden_size=2)
    assert lstm.trace_kwargs == {'input_size': 2, 'hidden_size': 2}


def test_subclass():
    @nni.trace
    class Super:
        def __init__(self, a, b):
            self._a = a
            self._b = b

    class Sub1(Super):
        def __init__(self, c, d):
            super().__init__(3, 4)
            self._c = c
            self._d = d

    @nni.trace
    class Sub2(Super):
        def __init__(self, c, d):
            super().__init__(3, 4)
            self._c = c
            self._d = d

    obj = Sub1(1, 2)
    # There could be trace_kwargs for obj. Behavior is undefined.
    assert obj._a == 3 and obj._c == 1
    assert isinstance(obj, Super)
    obj = Sub2(1, 2)
    assert obj.trace_kwargs == {'c': 1, 'd': 2}
    assert issubclass(type(obj), Super)
    assert isinstance(obj, Super)


class ConsistencyTest1:
    pass


class ConsistencyTest2:
    def __init__(self):
        self.test = nni.trace(ConsistencyTest1)()


def test_dump_consistency():
    test2 = ConsistencyTest2()
    symbol1 = test2.test.trace_symbol
    pickle.dumps(test2)
    symbol2 = test2.test.trace_symbol
    assert symbol1 == symbol2


def test_get():
    @nni.trace
    class Foo:
        def __init__(self, a = 1):
            self._a = a

        def bar(self):
            return self._a + 1

    obj = Foo(3)
    assert nni.load(nni.dump(obj)).bar() == 4
    obj1 = obj.trace_copy()
    with pytest.raises(AttributeError):
        obj1.bar()
    obj1.trace_kwargs['a'] = 5
    obj1 = obj1.get()
    assert obj1.bar() == 6
    obj2 = obj1.trace_copy()
    obj2.trace_kwargs['a'] = -1
    assert obj2.get().bar() == 0


class CustomParameter:
    def __init__(self, x):
        self._wrapped = x

    def _unwrap_parameter(self):
        return self._wrapped


def test_unwrap_parameter():
    c = CustomParameter(1)
    cls = SimpleClass(c)
    assert cls._a == 1
    assert cls.trace_kwargs['a'] == c
