import torch
import torch.nn.functional as F
from torch import nn

import pytest
import nni
import nni.nas.profiler.pytorch.utils.shape as _shape
from nni.mutable import MutableList, Categorical
from nni.nas.nn.pytorch import (
    MutableLinear, MutableConv2d, MutableDropout, MutableMultiheadAttention,
    ModelSpace, LayerChoice, InputChoice, Repeat
)
from nni.nas.profiler.pytorch.utils import shape_inference, submodule_input_output_shapes, ShapeTensor, MutableShape, profiler_leaf_module


def randn_tensor(*shape):
    default_shape = tuple(MutableList(shape).default())
    tensor = ShapeTensor(torch.randn(*default_shape), shape)
    return tensor


def test_assign_shape_info():
    from torch.utils._pytree import tree_map
    cases = [
        (
            ShapeTensor(torch.randn(3)),
            MutableShape(3),
        ),
        (
            (ShapeTensor(torch.randn(3, 4)), ShapeTensor(torch.randn(4, 5))),
            (MutableShape(3, 4), MutableShape(4, 5)),
        ),
        (
            [()],
            [()],
        ),
        (
            ([],),
            ([],),
        ),
        (
            {'a': ()},
            {'a': ()},
        ),
        (
            {'a': 0, 'b': [{'c': 1}]},
            {'a': 0, 'b': [{'c': 1}]},
        ),
        (
            {'a': 0, 'b': [1, {'c': 2}, ShapeTensor(torch.randn(3))], 'c': (ShapeTensor(torch.randn(2, 3)), 1)},
            {'a': 0, 'b': [1, {'c': 2}, MutableShape(3)], 'c': (MutableShape(2, 3), 1)},
        )
    ]

    for tensors, shapes in cases:
        _shape._assign_shape_info(tensors, shapes)

        def _check_shape(obj):
            if isinstance(obj, ShapeTensor):
                assert obj.real_shape is not None

        tree_map(_check_shape, tensors)


def test_switch_case_shape_info():
    shape = _shape.switch_case_shape_info(nni.choice('x', [1, 2, 3]), {
        1: MutableShape(2, 3),
        2: MutableShape(4, 5),
        3: MutableShape(6, 7),
    })
    assert isinstance(shape, MutableShape)
    assert shape[1].freeze({'x': 2}) == 5

    shape = _shape.switch_case_shape_info(nni.choice('x', [2, 4, 6]), {
        2: (MutableShape(2, 3), MutableShape(4, 5), 't'),
        4: (MutableShape(5, 3), MutableShape(7, 9), 'x'),
        6: (MutableShape(3, 3), MutableShape(4, 4), 'y'),
    })
    assert isinstance(shape, tuple) and len(shape) == 3
    assert isinstance(shape[0], MutableShape)
    assert shape[0][1] == 3
    assert shape[0].freeze({'x': 2}) == (2, 3)
    assert shape[1].freeze({'x': 4}) == (7, 9)
    assert shape[2] == 't'


def test_mutable_shape():
    shape = MutableShape(2, 3)
    assert shape != (2, 3)
    assert shape == MutableShape(2, 3)
    assert shape != MutableShape(2, 4)
    assert repr(shape) == 'MutableShape(2, 3)'
    assert not shape.is_mutable()

    shape = MutableShape()
    assert len(shape) == 0
    assert shape.numel() == 1
    assert not shape.is_mutable()

    with pytest.raises(TypeError):
        MutableShape(2.5, 3)

    with pytest.raises(TypeError):
        MutableShape(2, 3, nni.choice('x', [2.5, 3]))

    shape = MutableShape(2, 3, nni.choice('x', [2, 3, 4]))
    assert shape.default() == (2, 3, 2)
    assert shape == MutableShape([2, 3, nni.choice('x', [2, 3, 4])])
    assert len(list(shape.grid())) == 3
    assert shape.freeze({'x': 3}) == (2, 3, 3)
    assert list(shape.numel().grid()) == [12, 18, 24]
    assert shape.is_mutable()

    assert shape[1] == 3
    assert shape[2].values == [2, 3, 4]
    assert len(shape) == 3
    assert shape[:2] == MutableShape(2, 3)


def test_int_proxy():
    assert _shape.IntProxy(2, 2) == 2
    assert _shape.IntProxy(2, nni.choice('x', [2, 3])) == 2
    assert repr(_shape.IntProxy(2, 2)) == 'IntProxy(2, 2)'

    expr = _shape.IntProxy(2, nni.choice('x', [2, 3])) + _shape.IntProxy(3, nni.choice('x', [2, 3]))
    assert expr == 5
    assert set(_shape.IntProxy.unwrap(expr).grid()) == set([4, 6])


def test_error_message(caplog):
    class Net(nn.Module):
        def forward(self, x):
            return torch.stft(x, 4, return_complex=True)

    input = ShapeTensor(torch.randn(10, 8), True)
    with pytest.raises(RuntimeError, match='Shape inference failed because no shape inference formula'):
        shape_inference(Net(), input)

    assert 'Shape information is not explicitly propagated when executing' in caplog.text
    assert "  - '' (type: " in caplog.text
    caplog.clear()

    _shape._current_module_names = []

    class Net1(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Net()

        def forward(self, x):
            return self.inner(x)

    input = ShapeTensor(torch.randn(10, 8), True)
    with pytest.raises(RuntimeError, match='Shape inference failed because no shape inference formula'):
        shape_inference(Net1(), input)

    assert 'Shape information is not explicitly propagated when executing' in caplog.text
    assert "  - 'inner' (type: " in caplog.text
    assert "  - '' (type: " in caplog.text

    _shape._current_module_names = []

    caplog.clear()

    Net1._shape_forward = lambda self, x: x

    with pytest.raises(RuntimeError, match='Shape inference failed because no shape inference formula'):
        submodule_input_output_shapes(Net1(), input)
    assert 'Shape information is not explicitly propagated when executing' in caplog.text
    assert "  - 'inner' (type: " in caplog.text

    # FIXME: We can't warn inplace operations.
    #        Someone please fix this.
    # caplog.clear()

    # _shape._current_module_names = []

    # class Net2(nn.Module):
    #     def forward(self, x):
    #         t =  x.zero_()
    #         print('forward', t)
    #         return t

    # result = shape_inference(Net2(), input)
    # assert result.real_shape == input.real_shape
    # assert 'inplace operation' in caplog.text


def test_view():
    class Net(ModelSpace):
        def forward(self, x):
            sz = x.size(-1)
            out = x.view(-1, sz)
            return out

    class Net2(ModelSpace):
        def forward(self, x):
            out = x.view(-1)
            return out

    class Net3(ModelSpace):
        def forward(self, x):
            out = x.view((2, 4))
            return out

    input = ShapeTensor(torch.randn(4, 2))
    input.real_shape = MutableShape(4, nni.choice('c', [2, 3]))
    output = shape_inference(Net(), input)
    assert list(output.real_shape.grid()) == [(4, 2), (4, 3)]

    output = shape_inference(Net2(), input)
    assert list(output.real_shape.grid()) == [(8,), (12,)]

    assert shape_inference(Net3(), ShapeTensor(torch.randn(4, 2), True)).real_shape == MutableShape(2, 4)


def test_tensor():
    t = ShapeTensor(torch.randn(4, 2), True)
    assert t.real_shape == MutableShape(4, 2)
    assert t.size(0) == 4
    assert t.size(1) == 2

    t = ShapeTensor(torch.randn(4, 2, 3))
    assert t.real_shape is None
    assert repr(t) == 'ShapeTensor(unknown)'
    t.real_shape = MutableShape(4, 2, nni.choice('c', [2, 3]))

    shape = t.shape
    assert shape[0] == 4
    assert shape[1] == 2
    assert shape[2] == 3
    assert isinstance(shape[2], _shape.IntProxy)
    assert shape[2].expression.values == [2, 3]

    assert repr(t) == "ShapeTensor([4, 2, Categorical([2, 3], label='c')])"


def test_reshape(caplog):
    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t.real_shape = MutableShape(4, nni.choice('c', [2, 3]))

    t1 = torch.flatten(t, 1)
    assert t1.real_shape is None
    assert 'RuntimeError' in caplog.text

    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t = torch.flatten(t, 1)
    assert t.real_shape == MutableShape(4, 10)

    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t = t.reshape(2, 4, 5)
    assert t.real_shape == MutableShape(2, 4, 5)

    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t.real_shape = MutableShape(4, nni.choice('c', [2, 3]), 5)
    t = t.reshape(-1, 4, 5)
    assert t.real_shape.freeze({'c': 2}) == torch.Size([2, 4, 5])
    assert t.real_shape.freeze({'c': 3}) == torch.Size([3, 4, 5])

    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t.real_shape = MutableShape(4, nni.choice('c', [2, 3]), 5)
    a, b, c = t.size()
    t = t.reshape(c, a, b)
    assert t.real_shape == MutableShape(5, 4, nni.choice('c', [2, 3]))


def test_is_leaf():
    t = ShapeTensor(torch.randn(4, 2, 5), True)

    class SubModule(nn.Module):
        def forward(self, x):
            return ShapeTensor(torch.ones(4, 2, 5), False)

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = SubModule()

        def forward(self, x):
            return self.sub(x)

    with pytest.raises(RuntimeError, match='found for SubModule'):
        shape_inference(MyModule(), t)

    with pytest.raises(RuntimeError, match='found for MyModule'):
        shape_inference(MyModule(), t, is_leaf=lambda x: isinstance(x, MyModule))

    MyModule._shape_forward = lambda self, x: ShapeTensor(torch.ones(4, 2, 5), True)
    assert shape_inference(MyModule(), t, is_leaf=lambda x: isinstance(x, MyModule)).real_shape == MutableShape(4, 2, 5)


def test_mean():
    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t1 = t.mean(0)
    assert t1.real_shape == MutableShape(2, 5)
    t2 = t.mean((1, 2))
    assert t2.real_shape == MutableShape(4)

    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t.real_shape = MutableShape(4, nni.choice('c', [2, 3]), 5)
    t1 = t.mean(2)
    assert t1.real_shape == MutableShape(4, nni.choice('c', [2, 3]))
    t2 = t.mean(1, keepdim=True)
    assert t2.real_shape == MutableShape(4, 1, 5)

    t3 = t.mean(-1)
    assert t3.real_shape == MutableShape(4, nni.choice('c', [2, 3]))


def test_add_mul(caplog):
    t1 = ShapeTensor(torch.randn(4, 2, 5), True)
    t2 = ShapeTensor(torch.randn(4, 2, 5), True)
    assert (t1 + t2).real_shape == MutableShape(4, 2, 5)

    t1 = ShapeTensor(torch.randn(5, 3, 4, 1), True)
    t2 = ShapeTensor(torch.randn(3, 1, 1), True)
    assert (t1 + t2).real_shape == MutableShape(5, 3, 4, 1)
    assert (t1 * t2).real_shape == MutableShape(5, 3, 4, 1)

    t1 = ShapeTensor(torch.randn(4, 2, 5), True)
    t2 = ShapeTensor(torch.randn(4, 2, 5), True)
    t1.real_shape = MutableShape(4, nni.choice('c', [2, 3]), 5)
    # Theoretically it doesn't work. But we don't care.
    assert (t1 + t2).real_shape == MutableShape(4, nni.choice('c', [2, 3]), 5)

    t1 = ShapeTensor(torch.randn(4, 2, 5), True)
    t2 = ShapeTensor(torch.randn(4, 2, 5), False)
    assert (t1 + t2).real_shape == MutableShape(4, 2, 5)

    caplog.clear()
    t1 = ShapeTensor(torch.randn(4, 2, 5), False)
    t2 = ShapeTensor(torch.randn(4, 2, 5), False)
    assert (t1 + t2).real_shape is None
    assert 'failed though shape inference formula found' in caplog.text


def test_slice_select():
    t1 = ShapeTensor(torch.randn(4, 2, 5), True)
    t2 = t1[:, 1, 2:4]
    assert t2.real_shape == MutableShape(4, 2)

    t2 = t1[1:-1, :1, 3:6]
    assert t2.real_shape == MutableShape(2, 1, 2)

    t1 = ShapeTensor(torch.randn(4, 2, 5), True)
    t1.real_shape = MutableShape(4, nni.choice('c', [2, 3]), 5)
    t2 = t1[:, 1:]
    assert t2.real_shape.freeze({'c': 2}) == (4, 1, 5)
    assert t2.real_shape.freeze({'c': 3}) == (4, 2, 5)

    t2 = t1[0]
    assert t2.real_shape == MutableShape(nni.choice('c', [2, 3]), 5)


def test_permute():
    t = ShapeTensor(torch.randn(4, 2, 5), True)
    assert t.permute(1, 0, 2).real_shape == MutableShape(2, 4, 5)

    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t.real_shape = MutableShape(4, nni.choice('c', [2, 3]), 5)
    assert t.permute(1, 0, 2).real_shape.freeze({'c': 2}) == (2, 4, 5)
    assert t.permute(2, 1, 0).real_shape.freeze({'c': 3}) == (5, 3, 4)


def test_cat():
    t = ShapeTensor(torch.randn(4, 2, 5), True)
    t1 = ShapeTensor(torch.randn(4, 3, 5), True)
    t1.real_shape = MutableShape(4, nni.choice('c', [2, 3]), 5)
    assert torch.cat([t, t], 0).real_shape == MutableShape(8, 2, 5)
    assert torch.cat([t, t], -1).real_shape == MutableShape(4, 2, 10)
    assert torch.cat([t, t, t1], 1).real_shape == MutableShape(4, 4 + nni.choice('c', [2, 3]), 5)


def test_adaptive_avg_pool2d():
    t = ShapeTensor(torch.randn(4, 2, 5, 5), True)
    assert shape_inference(nn.AdaptiveAvgPool2d(1), t).real_shape == MutableShape(4, 2, 1, 1)
    assert shape_inference(nn.AdaptiveAvgPool2d(3), t).real_shape == MutableShape(4, 2, 3, 3)
    assert shape_inference(nn.AdaptiveAvgPool2d((3, 4)), t).real_shape == MutableShape(4, 2, 3, 4)

def test_avg_pool2d():
    t = ShapeTensor(torch.randn(4, 2, 5, 5), True)
    assert shape_inference(nn.AvgPool2d(1), t).real_shape == MutableShape(4, 2, 5, 5)
    assert shape_inference(nn.AvgPool2d(3,stride=1), t).real_shape == MutableShape(4, 2, 3, 3)
    assert shape_inference(nn.AvgPool2d((3, 4),stride=1), t).real_shape == MutableShape(4, 2, 3, 2)


def test_linear():
    input = ShapeTensor(torch.randn(4, 2), True)
    assert shape_inference(nn.Linear(2, 3), input).real_shape == MutableShape(4, 3)
    assert shape_inference(nn.Linear(2, 3), input).shape == (4, 3)
    assert shape_inference(MutableLinear(2, nni.choice('c', [2, 3])), input).real_shape == MutableShape(4, nni.choice('c', [2, 3]))

    assert shape_inference(nn.Sequential(
        MutableLinear(2, nni.choice('x', [3, 4])),
        MutableLinear(nni.choice('x', [3, 4]), 5)
    ), input).real_shape == MutableShape(4, 5)


def test_conv2d():
    input = ShapeTensor(torch.randn(4, 2, 8, 8), True)
    assert shape_inference(nn.Conv2d(2, 3, 3), input).real_shape == MutableShape(4, 3, 6, 6)

    kernel_size = nni.choice('k', [3, 5, 7])
    conv2d = MutableConv2d(2, nni.choice('c', [3, 4]), kernel_size, padding=kernel_size // 2)
    output = shape_inference(conv2d, input)
    assert output.real_shape.freeze({'c': 3, 'k': 5}) == (4, 3, 8, 8)
    assert output.real_shape.freeze({'c': 4, 'k': 7}) == (4, 4, 8, 8)
    assert output.real_shape.freeze({'c': 3, 'k': 3}) == (4, 3, 8, 8)


def test_mhattn():
    mhattn = nn.MultiheadAttention(6, 3)

    query = ShapeTensor(torch.randn(8, 2, 6), True)
    key = ShapeTensor(torch.randn(7, 2, 6), True)
    value = ShapeTensor(torch.randn(7, 2, 6), True)
    rv = shape_inference(mhattn, query, key, value)
    assert rv[0].real_shape == MutableShape(8, 2, 6)
    assert rv[1].real_shape == MutableShape(2, 8, 7)
    rv = shape_inference(mhattn, query, key, value, average_attn_weights=False)
    assert rv[1].real_shape == MutableShape(2, 3, 8, 7)

    query = ShapeTensor(torch.randn(8, 6), True)
    key = ShapeTensor(torch.randn(7, 6), True)
    value = ShapeTensor(torch.randn(7, 6), True)
    rv = shape_inference(mhattn, query, key, value)
    assert rv[0].real_shape == MutableShape(8, 6)
    assert rv[1].real_shape == MutableShape(8, 7)
    rv = shape_inference(mhattn, query, key, value, need_weights=False)
    assert rv[1] is None
    rv = shape_inference(mhattn, query, key, value, average_attn_weights=False)
    assert rv[1].real_shape == MutableShape(3, 8, 7)

    mhattn = nn.MultiheadAttention(6, 3, batch_first=True)

    query = ShapeTensor(torch.randn(2, 8, 6), True)
    key = ShapeTensor(torch.randn(2, 7, 6), True)
    value = ShapeTensor(torch.randn(2, 7, 6), True)
    rv = shape_inference(mhattn, query, key, value)
    assert rv[0].real_shape == MutableShape(2, 8, 6)
    assert rv[1].real_shape == MutableShape(2, 8, 7)
    rv = shape_inference(mhattn, query, key, value, average_attn_weights=False)
    assert rv[1].real_shape == MutableShape(2, 3, 8, 7)

    mhattn = MutableMultiheadAttention(nni.choice('a', [6, 12]), nni.choice('b', [2, 3]), batch_first=True)
    query = ShapeTensor(torch.randn(2, 8, 6), True)
    key = ShapeTensor(torch.randn(2, 7, 6), True)
    value = ShapeTensor(torch.randn(2, 7, 6), True)
    query.real_shape = MutableShape(2, 8, nni.choice('a', [6, 12]))
    key.real_shape = MutableShape(2, 7, nni.choice('a', [6, 12]))
    value.real_shape = MutableShape(2, 7, nni.choice('a', [6, 12]))
    rv = shape_inference(mhattn, query, key, value, average_attn_weights=False)
    assert rv[0].real_shape == MutableShape(2, 8, nni.choice('a', [6, 12]))
    assert rv[1].real_shape == MutableShape(2, nni.choice('b', [2, 3]), 8, 7)


def test_layer_choice():
    input = ShapeTensor(torch.randn(4, 2), True)
    layer_choice = LayerChoice([
        nn.Linear(2, 3),
        nn.Linear(2, 4)
    ], label='linear')
    shape = shape_inference(layer_choice, input).real_shape
    assert shape.freeze({'linear': 0}) == (4, 3)
    assert shape.freeze({'linear': 1}) == (4, 4)


def test_input_choice():
    net = InputChoice(3, 1, label='a')
    x = [ShapeTensor(torch.randn(3)) for _ in range(3)]
    for t in x:
        t.real_shape = MutableShape(nni.choice('b', [3, 4, 5]))
    result = shape_inference(net, x)
    assert result.real_shape == MutableShape(nni.choice('b', [3, 4, 5]))

    with pytest.raises(ValueError, match='multiple choices'):
        shape_inference(InputChoice(3, None), x)


def test_layer_choice_custom_return_type():
    class CustomModule(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            x, y = x
            return self.linear(x), self.linear(x) + self.linear(y), {'a': 1, 'b': torch.abs(self.linear(x))}

        def _shape_forward(self, x):
            x, y = x
            return shape_inference(self.linear, x), shape_inference(self.linear, y), {'a': 1, 'b': shape_inference(self.linear, x)}

    input1 = ShapeTensor(torch.randn(4, 2), True)
    input2 = ShapeTensor(torch.randn(4, 2), True)
    layer_choice = LayerChoice({
        'a': CustomModule(2, 3),
        'b': CustomModule(2, 4),
        'c': CustomModule(2, 5)
    }, label='linear')

    shape = _shape.extract_shape_info(shape_inference(layer_choice, (input1, input2)))
    assert shape[0].freeze({'linear': 'a'}) == (4, 3)
    assert shape[1].freeze({'linear': 'b'}) == (4, 4)
    assert isinstance(shape[2], dict)
    assert shape[2]['a'] == 1
    assert shape[2]['b'].freeze({'linear': 'c'}) == (4, 5)


def test_layer_choice_propagate_is_leaf():
    class Poisonous(nn.Module):
        def forward(self, x):
            return x

        def _shape_forward(self, x):
            raise RuntimeError('should not be called')

    class InnerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.poisonous = Poisonous()

        def forward(self, x):
            return self.poisonous(x)

        def _shape_forward(self, x):
            return x.real_shape

    class OuterNet(ModelSpace):
        def __init__(self):
            super().__init__()
            self.choice = LayerChoice([
                InnerNet(),
                InnerNet(),
            ], label='choice')

        def forward(self, x):
            return self.choice(x)

    with pytest.raises(RuntimeError, match='should not be called'):
        submodule_input_output_shapes(OuterNet(), ShapeTensor(torch.randn(3), True))

    def is_leaf(module):
        return isinstance(module, InnerNet)

    results = submodule_input_output_shapes(OuterNet(), ShapeTensor(torch.randn(3), True), is_leaf=is_leaf)
    assert set(results.keys()) == {'', 'choice', 'choice.0', 'choice.1'}


def test_repeat():
    input = ShapeTensor(torch.randn(4, 1), True)
    repeat = Repeat(lambda index: nn.Linear(index + 1, index + 2), 5)
    assert shape_inference(repeat, input).real_shape == MutableShape(4, 6)

    repeat = Repeat(lambda index: nn.Linear(index + 1, index + 2), nni.choice('n', [0, 1, 3, 4]))
    shape = shape_inference(repeat, input).real_shape
    assert shape.freeze({'n': 0}) == (4, 1)
    assert shape.freeze({'n': 1}) == (4, 2)
    assert shape.freeze({'n': 3}) == (4, 4)
    assert shape.freeze({'n': 4}) == (4, 5)

    repeat = Repeat(lambda index: MutableLinear((index + 1) * nni.choice('m', [1, 2]), (index + 2) * nni.choice('m', [1, 2])),
                    nni.choice('n', [0, 1, 3, 4]) + 2)
    shape = shape_inference(repeat, input).real_shape
    assert shape.freeze({'n': 0, 'm': 1}) == (4, 3)
    assert shape.freeze({'n': 1, 'm': 2}) == (4, 8)
    assert shape.freeze({'n': 3, 'm': 1}) == (4, 6)
    assert shape.freeze({'n': 4, 'm': 2}) == (4, 14)


def test_repeat_propagate_is_leaf():
    class Poisonous(nn.Module):
        def forward(self, x):
            return x

        def _shape_forward(self, x):
            raise RuntimeError('should not be called')

    class InnerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.poisonous = Poisonous()

        def forward(self, x):
            return self.poisonous(x)

        def _shape_forward(self, x):
            return x.real_shape

    class OuterNet(ModelSpace):
        def __init__(self):
            super().__init__()
            self.choice = Repeat(InnerNet(), (1, 2), label='rep')

        def forward(self, x):
            return self.choice(x)

    with pytest.raises(RuntimeError, match='should not be called'):
        submodule_input_output_shapes(OuterNet(), ShapeTensor(torch.randn(3), True))

    def is_leaf(module):
        return isinstance(module, InnerNet)

    results = submodule_input_output_shapes(OuterNet(), ShapeTensor(torch.randn(3), True), is_leaf=is_leaf)
    assert set(results.keys()) == {'', 'choice', 'choice.blocks.0', 'choice.blocks.1'}

    InnerNet = profiler_leaf_module(InnerNet)
    results = submodule_input_output_shapes(OuterNet(), ShapeTensor(torch.randn(3), True))
    assert set(results.keys()) == {'', 'choice', 'choice.blocks.0', 'choice.blocks.1'}


def test_mnist_model():
    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
            self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.pointwise(self.depthwise(x))

    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = LayerChoice([
                nn.Conv2d(32, 64, 3, 1),
                DepthwiseSeparableConv(32, 64)
            ], label='conv')
            self.pool = nn.MaxPool2d(2)
            self.dropout1 = MutableDropout(nni.choice('dropout', [0.25, 0.5, 0.75]))
            self.dropout2 = nn.Dropout(0.5)
            feature = nni.choice('feature', [64, 128, 256])
            self.fc1 = MutableLinear(9216, feature)
            self.fc2 = MutableLinear(feature, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = self.dropout1(x).view(x.size(0), -1)
            x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
            return x

    input = ShapeTensor(torch.randn(4, 1, 28, 28), True)
    output = shape_inference(Net(), input)
    assert output.real_shape == MutableShape(4, 10)

    net = Net()
    shapes = submodule_input_output_shapes(net, input)
    input_shapes = {k: v[0][0] for k, v in shapes.items()}  # unwrap tuple
    output_shapes = {k: v[1] for k, v in shapes.items()}
    expected_input_shapes = {
        '': MutableShape(4, 1, 28, 28),
        'conv1': MutableShape(4, 1, 28, 28),
        'conv2': MutableShape(4, 32, 26, 26),
        'conv2.0': MutableShape(4, 32, 26, 26),
        'conv2.1': MutableShape(4, 32, 26, 26),
        'conv2.1.depthwise': MutableShape(4, 32, 26, 26),
        'conv2.1.pointwise': MutableShape(4, 32, 24, 24),
        'dropout1': MutableShape(4, 64, 12, 12),
        'dropout2': MutableShape(4, Categorical([64, 128, 256], label=net.fc1.args['out_features'].label)),
        'fc1': MutableShape(4, 9216),
        'fc2': MutableShape(4, Categorical([64, 128, 256], label=net.fc1.args['out_features'].label)),
        'pool': MutableShape(4, 64, 24, 24)
    }
    expected_output_shapes = {
        '': MutableShape(4, 10),
        'conv1': MutableShape(4, 32, 26, 26),
        'conv2': MutableShape(4, 64, 24, 24),
        'conv2.0': MutableShape(4, 64, 24, 24),
        'conv2.1': MutableShape(4, 64, 24, 24),
        'conv2.1.depthwise': MutableShape(4, 32, 24, 24),
        'conv2.1.pointwise': MutableShape(4, 64, 24, 24),
        'dropout1': MutableShape(4, 64, 12, 12),
        'dropout2': MutableShape(4, Categorical([64, 128, 256], label=net.fc1.args['out_features'].label)),
        'fc1': MutableShape(4, Categorical([64, 128, 256], label=net.fc1.args['out_features'].label)),
        'fc2': MutableShape(4, 10),
        'pool': MutableShape(4, 64, 12, 12)
    }
    assert input_shapes == expected_input_shapes
    assert output_shapes == expected_output_shapes
