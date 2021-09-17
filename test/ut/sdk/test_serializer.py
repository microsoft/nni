import nni
import torch


@nni.trace
class SimpleClass:
    def __init__(self, a, b=1):
        self._a = a
        self._b = b



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
        r'{"__symbol__": "path:torch.nn.modules.conv", ' \
        r'"__kwargs__": {"in_channels": 3, "out_channels": 16, "kernel_size": 3}}'

    print(nni.load(nni.dump(conv)))


if __name__ == '__main__':
    test_simple_class()
    test_external_class()