import nni
import torch


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
    assert instance.get()._a == 1
    assert instance.get()._b == 2


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
    assert conv.get().kernel_size == (3, 3)


def test_nested_class():
    a = SimpleClass(1, 2)
    b = SimpleClass(a)
    assert b._a._a == 1
    dump_str = nni.dump(b)
    b = nni.load(dump_str)
    assert repr(b) == 'SerializableObject(type=SimpleClass, a=SerializableObject(type=SimpleClass, a=1, b=2))'
    assert b.get()._a._a == 1


def test_unserializable():
    a = UnserializableSimpleClass()
    dump_str = nni.dump(a)
    a = nni.load(dump_str)
    assert a._a == 1


if __name__ == '__main__':
    test_simple_class()
    test_external_class()
    test_nested_class()
    test_unserializable()
