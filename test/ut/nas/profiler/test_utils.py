import torch
from torch import nn

import nni
import nni.nas.nn.pytorch as nas_nn
from nni.mutable import Categorical, frozen_context
from nni.mutable.mutable import _mutable_equal
from nni.nas.profiler.pytorch.utils import _attrs, is_leaf_module, _expression


def test_is_leaf_module():
    assert is_leaf_module(nn.Linear(2, 3))
    assert not is_leaf_module(nn.Sequential(nn.Linear(2, 3)))
    assert is_leaf_module(nas_nn.MutableLinear(2, 3))

    class MyLinear(nn.Linear):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.register_buffer('buf', torch.randn(2, 3))

    assert not is_leaf_module(MyLinear(2, 3))


def test_getitem():
    assert _attrs._getitem([1, 2, 3], 1) == 2
    assert _attrs._getitem(2, 0) == 2
    expr = _attrs._getitem(Categorical([3, (4, 5), 6], label='a'), 1)
    assert expr.freeze({'a': 3}) == 3
    assert expr.freeze({'a': (4, 5)}) == 5
    assert expr.freeze({'a': 6}) == 6


def test_getattr():
    conv2d = nn.Conv2d(1, 3, 2, 2, 0, 1)
    assert _attrs._getattr(conv2d, 'in_channels') == 1
    assert _attrs._getattr(conv2d, 'out_channels') == 3
    assert _attrs._getattr(conv2d, 'kernel_size') == (2, 2)
    assert _attrs._getattr(conv2d, 'kernel_size', _attrs.tuple_2_t) == (2, 2)
    assert _attrs._getattr(conv2d, 'kernel_size', _attrs.tuple_n_t[2]) == (2, 2)
    assert _attrs._getattr(conv2d, 'stride') == (2, 2)
    assert _attrs._getattr(conv2d, 'bias', bool) == True
    assert _attrs._getattr(conv2d, 'bias', int) == 1

    assert _attrs._getattr(nn.Conv2d(1, 3, 1, bias=False), 'bias', bool) == False

    conv2d = nas_nn.MutableConv2d(1, 3, 2, 2, 0, 1)
    assert _attrs._getattr(conv2d, 'kernel_size') == 2
    assert _attrs._getattr(conv2d, 'kernel_size', _attrs.tuple_2_t) == (2, 2)

    conv2d = nas_nn.MutableConv2d(1, 3, (1, 2), 2, 0, 1)
    assert _attrs._getattr(conv2d, 'kernel_size', _attrs.tuple_2_t) == (1, 2)

    with frozen_context():
        conv2d = nas_nn.MutableConv2d(1, 3, Categorical([(1, 2), (2, 1), 3], label='a'), (2, 3), 0, 1)
    kernel_size = _attrs._getattr(conv2d, 'kernel_size', _attrs.tuple_2_t)
    assert kernel_size[0].freeze({'a': (1, 2)}) == 1
    assert kernel_size[1].freeze({'a': (1, 2)}) == 2
    assert kernel_size[0].freeze({'a': 3}) == 3
    assert kernel_size[1].freeze({'a': 3}) == 3

    assert _attrs._getattr(conv2d, 'bias', bool) == True

    with frozen_context():
        conv1d = nas_nn.MutableConv1d(1, 3, 1, bias=Categorical([True, False], label='b'))
    assert _attrs._getattr(conv1d, 'bias', bool).freeze({'b': False}) == False


def test_conclude_assumption():
    assert _expression.conclude_assumptions([1, 2, 3]) == {
        'real': True, 'integer': True, 'positive': True, 'nonnegative': True, 'nonzero': True
    }
    assert _expression.conclude_assumptions([0, 0, 0]) == {
        'real': True, 'integer': True, 'zero': True,
        'even': True, 'nonnegative': True, 'nonpositive': True
    }
    assert _expression.conclude_assumptions([-1, -2, -3]) == {
        'real': True, 'integer': True, 'negative': True, 'nonpositive': True, 'nonzero': True
    }
    assert _expression.conclude_assumptions([1, 3, 5, -1]) == {
        'real': True, 'integer': True, 'odd': True, 'nonzero': True
    }
    assert _expression.conclude_assumptions([2, 4, 6, -2, 0]) == {
        'real': True, 'integer': True, 'even': True
    }
    assert _expression.conclude_assumptions([1.0, 2.0, 3.0]) == {'integer': False, 'real': True, 'nonnegative': True, 'nonzero': True, 'positive': True}
    assert _expression.conclude_assumptions([1.0, 2, 3]) == {'integer': False, 'real': True, 'nonnegative': True, 'nonzero': True, 'positive': True}
    assert _expression.conclude_assumptions([1.0, 2.0, 3]) == {'integer': False, 'real': True, 'nonnegative': True, 'nonzero': True, 'positive': True}
    assert _expression.conclude_assumptions([1, 2.0, 3]) == {'integer': False, 'real': True, 'nonnegative': True, 'nonzero': True, 'positive': True}
    assert _expression.conclude_assumptions(['cat', 'dog']) == {'real': False}


def test_expression_simplification():
    simp = _expression.expression_simplification

    assert _mutable_equal(
        simp(nni.choice('a', [2, 3]) + nni.choice('b', [3, 4]) - nni.choice('a', [2, 3])),
        nni.choice('b', [3, 4])
    )

    x = nni.choice('x', [32, 64])
    y = nni.choice('y', [3, 5, 7])
    assert _mutable_equal(
        simp(x + 2 * ((y - 1) // 2) - (y - 1) - 1),
        x - 1
    )

    assert _mutable_equal(simp(x - (x + 1)), -1)
    assert isinstance(simp(x - (x + 1)), int)

    x = nni.choice('x', [0., 1., 2.])
    assert _mutable_equal(simp((x + 1) ** 2 - (x * x + 2 * x + 2)), -1)
    assert isinstance(simp(x - (x + 1)), float)

    x = nni.choice('x', [2, 4, 8])
    assert _mutable_equal(simp(x // 2 - x / 2), 0)
    assert isinstance(simp(x // 2 - x / 2), float)

    x = nni.choice('x', [3, 5, 7])
    expr = -x // 2 + (x - 1) // 2 + 1
    assert simp(expr) == 0


def test_recursive_simplification():
    from nni.nas.profiler.pytorch.utils import MutableShape

    x = round(nni.choice('x', [32, 64]) / 2)
    y = nni.choice('y', [3, 5, 7]) - nni.choice('y', [3, 5, 7]) + 1
    z = nni.choice('y', [3, 5, 7]) + nni.choice('y', [3, 5, 7]) - nni.choice('x', [32, 64])
    shape = MutableShape(x, y, z)
    d = {
        'y': y,
        'w': (y, z),
        'v': [x, y, z],
        'shape': shape
    }

    simplified_z = _expression.expression_simplification(z)
    assert sorted(z.grid()) == sorted(simplified_z.grid())

    assert _mutable_equal(
        _expression.recursive_simplification(d),
        {
            'y': 1,
            'w': (1, simplified_z),
            'v': [x, 1, simplified_z],
            'shape': MutableShape(x, 1, simplified_z)
        }
    )
