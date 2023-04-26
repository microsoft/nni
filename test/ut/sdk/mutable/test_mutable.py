import math
from collections import Counter

import numpy as np
import pytest

from nni.mutable import *
from nni.mutable.mutable import _dedup_labeled_mutables, _mutable_equal


def test_symbolic_execution():
    a = Categorical([1, 2], label='a')
    b = Categorical([3, 4], label='b')
    c = Categorical([5, 6], label='c')
    d = a + b + 3 * c
    assert d.freeze({'a': 2, 'b': 3, 'c': 5}) == 20
    expect = [x + y + 3 * z for x in [1, 2] for y in [3, 4] for z in [5, 6]]
    assert list(d.grid()) == expect

    a = Categorical(['cat', 'dog'])
    b = Categorical(['milk', 'coffee'])
    assert (a + b).evaluate(['dog', 'coffee']) == 'dogcoffee'
    assert (a + 2 * b).evaluate(['cat', 'milk']) == 'catmilkmilk'

    assert (3 - Categorical([1, 2])).evaluate([1]) == 2

    with pytest.raises(
        TypeError,
        match=r'^can only concatenate str'
    ):
        (a + Categorical([1, 3])).default()

    a = Categorical([1, 17], label='aa')
    a = (abs(-a * 3) % 11) ** 5
    assert 'abs' in repr(a)
    with pytest.raises(
        SampleValidationError,
        match=r'^42 not found in'
    ):
        a.freeze({'aa': 42})
    assert a.evaluate([17]) == 7 ** 5

    a = round(7 / Categorical([2, 5]))
    assert a.evaluate([2]) == 4

    a = ~(77 ^ (Categorical([1, 4]) & 5))
    assert a.evaluate([4]) == ~(77 ^ (4 & 5))

    a = Categorical([5, 3]) * Categorical([6.5, 7.5])
    assert math.floor(a.evaluate([5, 7.5])) == int(5 * 7.5)

    a = Categorical([1, 3])
    b = Categorical([2, 4])
    with pytest.raises(
        RuntimeError,
        match=r'^Cannot use bool\(\) on SymbolicExpression'
    ):
        min(a, b)
    with pytest.raises(
        RuntimeError,
        match=r'^Cannot use bool\(\) on SymbolicExpression'
    ):
        if a < b:
            ...

    assert MutableExpression.min(a, b).evaluate([3, 2]) == 2
    assert MutableExpression.max(a, b).evaluate([3, 2]) == 3
    assert MutableExpression.max(1, 2, 3) == 3
    assert MutableExpression.max([1, 3, 2]) == 3

    assert MutableExpression.condition(Categorical([2, 3]) <= 2, 'a', 'b').evaluate([3]) == 'b'
    assert MutableExpression.condition(Categorical([2, 3]) <= 2, 'a', 'b').evaluate([2]) == 'a'

    with pytest.raises(RuntimeError):
        assert int(Categorical([2.5, 3.5])).evalute([2.5]) == 2

    assert MutableExpression.to_int(Categorical([2.5, 3.5])).evaluate([2.5]) == 2
    assert MutableExpression.to_float(Categorical(['2.5', '3.5'])).evaluate(['3.5']) == 3.5


def test_make_divisible():
    def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
        if min_value is None:
            min_value = divisor
        new_value = MutableExpression.max(min_value, MutableExpression.to_int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than (1-min_ratio).
        return MutableExpression.condition(new_value < min_ratio * value, new_value + divisor, new_value)

    def original_make_divisible(value, divisor, min_value=None, min_ratio=0.9):
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than (1-min_ratio).
        if new_value < min_ratio * value:
            new_value += divisor
        return new_value

    values = [4, 8, 16, 32, 64, 128]
    divisors = [2, 3, 5, 7, 15]
    with pytest.raises(
        RuntimeError,
        match=r'^(`__index__` is not allowed on SymbolicExpression|Try to use `SymbolicExpression.to_int)'
    ):
        original_make_divisible(Categorical(values, label='value'), Categorical(divisors, label='divisor'))
    result = make_divisible(Categorical(values, label='value'), Categorical(divisors, label='divisor'))
    for value in values:
        for divisor in divisors:
            lst = [value if choice.label == 'value' else divisor for choice in result.leaf_symbols()]
            assert result.evaluate(lst) == original_make_divisible(value, divisor)

            assert result.evaluate({'value': value, 'divisor': divisor}) == original_make_divisible(value, divisor)

    assert len(list(result.grid())) == 30
    assert max(result.grid()) == 135


def test_categorical():
    a = Categorical([1, 2, 3], label='a')
    assert a.simplify() == {'a': a}
    assert a.freeze({'a': 2}) == 2
    exception = a.check_contains({'a': 4})
    assert exception is not None and 'not found' in exception.msg
    assert a.contains({'a': 2})
    assert a.contains({'a': 3})
    assert not a.contains({'a': 4})
    with pytest.raises(
        SampleValidationError,
        match=r'^4 not found in'
    ):
        a.validate({'a': 4})

    with pytest.raises(AssertionError, match='must be unique'):
        Categorical([2, 2, 5])

    assert list(a.grid()) == [1, 2, 3]
    a = Categorical([1, 2, 3], weights=[0.2, 0.1, 0.7])
    assert list(a.grid()) == [3, 1, 2]

    counter = Counter()
    for _ in range(1000):
        counter[a.random()] += 1
    assert 120 <= counter[1] <= 280
    assert 50 <= counter[2] <= 150
    assert 500 <= counter[3] <= 900


def test_categorical_multiple():
    a = CategoricalMultiple([2, 3, 5], n_chosen=None, label='a')
    assert a.simplify() == {'a': a}
    a.freeze({'a': [2, 3]}) == [2, 3]

    def breakdown_dm(x): return isinstance(x, LabeledMutable) and not isinstance(x, CategoricalMultiple)

    s = a.simplify(is_leaf=breakdown_dm)
    assert len(s) == 3
    assert s['a/0'].values == [True, False]
    with pytest.raises(
        SampleValidationError,
        match=r'a/2 is missing'
    ):
        a.freeze({'a/0': False, 'a/1': True})
    assert a.freeze({'a/0': False, 'a/1': True, 'a/2': True}) == [3, 5]

    a = CategoricalMultiple([2, 3, 5], n_chosen=2, label='a')
    assert a.simplify() == {'a': a}
    assert len(a.random()) == 2 and a.random() in [[2, 3], [2, 5], [3, 5]]
    a.freeze({'a': [2, 3]}) == [2, 3]
    with pytest.raises(
        SampleValidationError,
        match=r'must have length 2'
    ):
        a.freeze({'a': [2, 3, 5]})

    s = a.simplify(is_leaf=breakdown_dm)
    assert len(s) == 4
    assert isinstance(s['a/n'], ExpressionConstraint)
    a.freeze({'a/0': True, 'a/1': False, 'a/2': True}) == [2, 5]
    with pytest.raises(
        SampleValidationError,
        match='is not satisfied'
    ):
        a.freeze({'a/0': False, 'a/1': False, 'a/2': True})

    a = CategoricalMultiple([1, 2, 3], weights=[0.2, 0.1, 0.7])
    assert list(a.grid()) == [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]

    counter = Counter()
    for _ in range(1000):
        for x in a.random():
            counter[x] += 1
    assert 120 <= counter[1] <= 280
    assert 50 <= counter[2] <= 150
    assert 500 <= counter[3] <= 900

    a = CategoricalMultiple([1, 2, 3], n_chosen=2, weights=[0.3, 0.1, 0.6])
    counter = Counter()
    for _ in range(1000):
        for x in a.random():
            counter[x] += 1
    assert counter[2] <= counter[1] <= counter[3]


def test_numerical():
    a = Numerical(0, 1, label='a')
    assert a.simplify() == {'a': a}
    assert a.freeze({'a': 0.5}) == 0.5
    exc = a.check_contains({'a': 4})
    assert exc is not None and 'higher than' in exc.msg
    assert a.contains({'a': 0.8})
    assert a.default() == 0.5
    assert 0 < a.random() < 1

    grid = a.grid()
    assert list(grid) == [0.5]

    grid = a.grid(granularity=2)
    assert list(grid) == [0.25, 0.5, 0.75]

    a = Numerical(0, 1, log_distributed=True, label='a')
    assert a.simplify() == {'a': a}

    a = Numerical(mu=0, sigma=1, label='a')
    assert -5 < a.random() < 5

    a = Numerical(mu=0, sigma=1, log_distributed=True, label='a')
    for _ in range(10):
        assert a.random() > 0

    a = Numerical(mu=0, sigma=1, low=-1, high=1)
    assert min(a.grid(granularity=4)) == -1

    a = Numerical(mu=2, sigma=3)
    x = [a.random() for _ in range(1000)]
    assert 0.5 < np.mean(x) < 2.5
    assert 2 < np.std(x) < 4
    assert np.mean(list(a.grid(granularity=4))) == 2

    a = Numerical(low=0, high=100, quantize=2)
    assert len(list(a.grid(granularity=10))) == 51
    assert a.random() % 2 == 0

    a = Numerical(low=1, high=3, quantize=0.75)
    assert len(list(a.grid(granularity=10))) == 4
    for x in a.grid(granularity=10):
        assert a.contains({a.label: x})

    a = Numerical(low=2, high=6, log_distributed=True, label='x')
    for _ in range(10):
        assert 2 < a.random() < 6
    with pytest.raises(
        SampleValidationError,
        match=r'than lower bound'
    ):
        a.freeze({'x': 1.5})

    from scipy.stats import beta
    a = Numerical(distribution=beta(2, 5), label='x')
    assert 0 < a.random() < 1
    assert 0.1 < a.default() < 0.3

    with pytest.raises(
        SampleValidationError,
        match=r'not in the distribution'
    ):
        a.freeze({'x': 1.5})


def test_mutable_list():
    a = MutableList([1, Categorical([1, 2, 3]), 3])
    assert a.default() == [1, 1, 3]
    a.append(Categorical([4, 5, 6]))
    assert a.default() == [1, 1, 3, 4]


def test_mutable_dict():
    a = MutableDict({
        'a': 1,
        'b': Categorical([1, 2, 3]),
        'c': 3
    })
    assert list(a.default().values()) == [1, 1, 3]
    assert list(a.default().keys()) == ['a', 'b', 'c']
    assert list(a.grid()) == [
        {'a': 1, 'b': 1, 'c': 3},
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 1, 'b': 3, 'c': 3},
    ]
    a.pop('b')
    assert list(a.grid()) == [
        {'a': 1, 'c': 3},
    ]
    assert a.random() == {'a': 1, 'c': 3}
    a['b'] = Numerical(0, 1)
    assert a.default() == {'a': 1, 'b': 0.5, 'c': 3}
    assert list(a.default().values()) == [1, 3, 0.5]

    search_space = MutableDict({
        'trainer': MutableDict({
            'optimizer': Categorical(['sgd', 'adam']),
            'learning_rate': Numerical(1e-4, 1e-2, log_distributed=True),
            'decay_epochs': MutableList([
                Categorical([10, 20]),
                Categorical([30, 50])
            ]),
        }),
        'model': MutableDict({
            'type': Categorical(['resnet18', 'resnet50']),
            'pretrained': Categorical([True, False])
        }),
    })

    assert len(search_space.random()) == 2
    assert len(list(search_space.grid(granularity=2))) == 96

    keys = list(search_space.simplify().keys())

    sample = search_space.freeze({
        keys[0]: 'adam',
        keys[1]: 0.0001,
        keys[2]: 10,
        keys[3]: 50,
        keys[4]: 'resnet50',
        keys[5]: False
    })
    assert sample['trainer']['decay_epochs'][1] == 50

    search_space = MutableList([
        MutableDict({
            'in_features': Categorical([10, 20], label='hidden_dim'),
            'out_features': Categorical([10, 20], label='hidden_dim') * 2,
        }),
        MutableDict({
            'in_features': Categorical([10, 20], label='hidden_dim') * 2,
            'out_features': Categorical([10, 20], label='hidden_dim') * 4,
        }),
    ])
    sample = search_space.default()
    assert sample[0]['out_features'] * 2 == sample[1]['out_features']


def test_composite():
    # Inspired by OpenAI gym:
    # https://github.com/openai/gym/blob/master/tests/spaces/utils.py
    COMPOSITE_SPACES = [
        MutableList([Categorical(range(5)), Categorical(range(4))]),
        MutableList([
            Categorical(range(5)),
            Numerical(0, 5)
        ]),
        MutableList([Categorical(range(5)), MutableList([Numerical(low=0.0, high=1.0), Categorical(range(2))])]),
        MutableList([Categorical(range(3)), MutableDict(
            position=Numerical(low=0.0, high=1.0),
            velocity=Categorical(range(2))
        )]),
        MutableDict(
            {
                'position': Categorical(range(5)),
                'velocity': Numerical(low=1, high=5, log_distributed=True),
            }
        ),
        MutableDict(
            position=Categorical(range(6)),
            velocity=Numerical(low=1, high=5, log_distributed=True),
        ),
        # TODO: Graph not supported yet.
        # MutableList((Graph(node_space=Box(-1, 1, shape=(2, 1)), edge_space=None), Categorical(2))),
        # MutableDict(
        #     a=MutableDict(
        #         a=Graph(node_space=Numerical(-100, 100), edge_space=None),
        #         b=Numerical(-100, 100),
        #     ),
        #     b=MutableList(Numerical(-100, 100), Numerical(-100, 100))
        # ),
        # Graph(node_space=Numerical(low=-100, high=100), edge_space=Categorical(range(5))),
        # Graph(node_space=Categorical(range(5)), edge_space=Numerical(low=-100, high=100)),
        # Graph(node_space=Categorical(3), edge_space=Categorical(range(4))),
    ]

    for space in COMPOSITE_SPACES:
        # Sanity check
        space.default()
        space.random()
        for _ in space.grid():
            pass

    space = MutableDict({
        'a': Numerical(low=0, high=1, label='x'),
        'b': MutableDict({
            'b_1': Numerical(low=-100, high=100),
            'b_2': Numerical(low=-1, high=1),
            'b_3': Numerical(low=0, high=1, label='x')
        }),
        'c': Categorical(range(4)),
    })

    for _ in range(10):
        sample = space.random()
        assert sample['a'] == sample['b']['b_3']


class MyCategorical(Categorical):
    def __hash__(self):
        return id(self)  # used in set


def test_dedup():
    a = Categorical([1, 2, 3], label='a')
    b = Categorical([1, 2, 3], label='a')
    assert a.equals(b)

    assert len(_dedup_labeled_mutables([a, b])) == 1

    b = Categorical([1, 2, 3, 4], label='a')
    with pytest.raises(ValueError, match='are different'):
        _dedup_labeled_mutables([a, b])

    b = MyCategorical([1, 2, 3], label='a')
    with pytest.raises(ValueError, match='are different'):
        _dedup_labeled_mutables([a, b])

    a = Numerical(0, 1, log_distributed=True, label='a')
    b = Numerical(0, 1, log_distributed=True, label='a')

    assert len(_dedup_labeled_mutables([a, b])) == 1
    assert not a.equals(Numerical(0, 1, log_distributed=False, label='a'))
    assert not a.equals(Numerical(mu=0, sigma=1, label='a'))

    a = Numerical(0, 1, label='a', default=0.5)
    b = Numerical(0, 1, label='a', default=0.3)
    assert not a.equals(b)


def test_is_leaf():
    a = Categorical([1, 2, 3], label='a')

    with pytest.raises(ValueError, match=r'is_leaf\(\) should return'):
        a.simplify(is_leaf=lambda x: False)


def test_repr():
    mutable = Mutable()
    assert repr(mutable) == 'Mutable()'

    categorical = Categorical([1, 2, 3], label='a')
    assert repr(categorical) == 'Categorical([1, 2, 3], label=\'a\')'

    categorical = Categorical(list(range(100)), label='a')
    assert repr(categorical) == 'Categorical([0, 1, 2, ..., 97, 98, 99], label=\'a\')'

    categorical = CategoricalMultiple([1, 2, 3], n_chosen=None, label='a')
    assert repr(categorical) == 'CategoricalMultiple([1, 2, 3], n_chosen=None, label=\'a\')'

    numerical = Numerical(0, 1, label='a')
    assert repr(numerical) == 'Numerical(0, 1, label=\'a\')'


def test_default():
    D = MutableDict({
        'a': Categorical([1, 2, 3], label='a'),
        'b': Categorical([4, 5, 6], label='b'),
        'c': MutableList([
            Categorical([1, 2, 3], label='a'),
            Numerical(0, 1, label='d'),
        ]),
        'd': Numerical(0, 1, label='d')
    })

    assert D.default() == {'a': 1, 'b': 4, 'c': [1, 0.5], 'd': 0.5}
    assert Categorical([1, 2, 3], default=2).default() == 2

    assert CategoricalMultiple([2, 4, 6], n_chosen=2).default() == [2, 4]
    assert CategoricalMultiple([5, 3, 7], n_chosen=None).default() == [5, 3, 7]

    with pytest.raises(ValueError, match='not a multiple of'):
        assert Numerical(0, 1, default=0.5, quantize=0.3).default() == 0.5

    assert Numerical(0, 1, default=0.9, quantize=0.3).default() == 0.9
    assert Numerical(0, 1, quantize=0.3).default() == 0.6

    x = Categorical([1, 2, 3], label='x')
    y = Categorical([4, 5, 6], label='y')
    exp = x + y
    assert exp.default() == 5

    with pytest.raises(ConstraintViolation):
        assert ExpressionConstraint(exp == 6).default() == None

    sample = {}
    assert ExpressionConstraint(exp == 6).robust_default(sample) is None
    assert sample['x'] + sample['y'] == 6
    assert x.default(sample) + y.default(sample) == 6

    assert ExpressionConstraint(exp == 6).robust_default(sample) is None
    with pytest.raises(ValueError, match=r'after \d+ retries'):
        ExpressionConstraint(exp == 7).robust_default(sample, retries=100)

    sample = {}
    assert ExpressionConstraint(exp == 7).robust_default(sample) is None
    assert sample['x'] + sample['y'] == 7

    with pytest.raises(ValueError, match=r'after \d+ retries'):
        ExpressionConstraint(exp == 10).robust_default(retries=100)

    with pytest.raises(ConstraintViolation):
        lst = MutableList([ExpressionConstraint(exp == 7), x, y]).default()
    lst = MutableList([ExpressionConstraint(exp == 7), x, y]).robust_default()
    assert lst[1] + lst[2] == 7

    # Specified default value conflicts with random sample
    x = Categorical([1, 2, 3], label='x', default=2)
    y = Categorical([4, 5, 6], label='y', default=4)
    sample = {}
    ExpressionConstraint(exp == 7).robust_default(sample)
    with pytest.raises(ValueError, match=r'Default value is specified to be'):
        x.default(sample)
        y.default(sample)


def test_random():
    lst = MutableList([
        Categorical([1, 2, 3]),
        Numerical(4, 6, label='z'),
        Numerical(4, 6, log_distributed=True),
        Numerical(mu=0, sigma=1),
        Numerical(4, 6, label='z')
    ])

    assert lst.random(random_state=np.random.RandomState(0)) == \
        lst.random(random_state=np.random.RandomState(0))
    sample = lst.random(random_state=np.random.RandomState(0))
    assert sample[1] == sample[4]

    x = Categorical([1, 2, 3], label='x', default=2)
    y = Categorical([4, 5, 6], label='y', default=4)
    with pytest.raises(ConstraintViolation):
        for _ in range(50):
            ExpressionConstraint(x + y == 7).random()


def test_grid():
    lst = MutableList([
        Categorical([1, 2, 3]),
        Numerical(4, 6, label='z'),
        Numerical(4, 6, log_distributed=True),
        Numerical(mu=0, sigma=1),
        Numerical(4, 6, label='z')
    ])

    assert len(list(lst.grid())) == 3

    x = Categorical([1, 2, 3], label='x')
    y = Categorical([4, 5, 6], label='y')
    exp = x + y
    assert len(list(ExpressionConstraint(exp == 7).grid())) == 3
    assert len(list(ExpressionConstraint(exp == 10).grid())) == 0

    assert list(MutableDict({
        'c': ExpressionConstraint(exp == 7),
        'a': x,
        'b': y
    }).grid()) == [
        {'c': None, 'a': 1, 'b': 6},
        {'c': None, 'a': 2, 'b': 5},
        {'c': None, 'a': 3, 'b': 4}
    ]

    lst = MutableList([1, 2, 3])
    assert list(lst.grid()) == [[1, 2, 3]]


def test_equals():
    assert _mutable_equal(Categorical([1, 2, 3], label='x'), Categorical([1, 2, 3], label='x'))
    assert not _mutable_equal(Categorical([1, 2, 3], label='x'), 1)
    assert not _mutable_equal(1, Categorical([1, 2, 3], label='x'))
    assert not _mutable_equal(1, 2)
    assert not _mutable_equal(False, True)
    assert _mutable_equal(False, False)
    assert _mutable_equal('a', 'a')
    assert _mutable_equal('abc', 'abc')
    assert not _mutable_equal('abc', 'abcd')
    assert _mutable_equal({
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='y'),
    }, {
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='y'),
    })
    assert not _mutable_equal({
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='y'),
    }, {
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='y'),
        'c': Categorical([7, 8, 9], label='z'),
    })
    assert not _mutable_equal({
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='y'),
    }, {
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='z'),
    })
    assert not _mutable_equal({
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='y'),
        'c': Categorical([7, 8, 9], label='z'),
    }, {
        'a': Categorical([1, 2, 3], label='x'),
        'b': Categorical([4, 5, 6], label='y'),
    })
    assert _mutable_equal({
        'a': ['b', 'c', Numerical(0, 1, label='x')],
    }, {
        'a': ['b', 'c', Numerical(0, 1, label='x')],
    })
    assert _mutable_equal(
        [1, 2, 3], [1, 2, 3]
    )
    assert not _mutable_equal(
        [1, 2, 3], [1, 2, 3, 4]
    )
    assert not _mutable_equal(
        [1, 2, Categorical([1, 2, 3])], [1, 2, 3]
    )

    assert _mutable_equal(
        {MyCategorical([0, 1], label='x'), MyCategorical([2, 3], label='y')},
        {MyCategorical([0, 1], label='x'), MyCategorical([2, 3], label='y')},
    )
    assert not _mutable_equal(
        {MyCategorical([0, 1], label='x'), MyCategorical([2, 3], label='y'), 3},
        {MyCategorical([0, 1], label='x'), MyCategorical([2, 3], label='y')},
    )
    assert not _mutable_equal(
        {MyCategorical([0, 1], label='x'), MyCategorical([4, 5], label='y')},
        {MyCategorical([0, 1], label='x'), MyCategorical([2, 3], label='y')},
    )

    assert _mutable_equal(
        Categorical([1, 2], label='a') * 0.75 + Numerical(0, 1, label='b'),
        Categorical([1, 2], label='a') * 0.75 + Numerical(0, 1, label='b'),
    )
    assert _mutable_equal(
        MutableExpression.to_int(Categorical([1, 2], label='a') * 0.75) + Numerical(0, 1, label='b'),
        MutableExpression.to_int(Categorical([1, 2], label='a') * 0.75) + Numerical(0, 1, label='b'),
    )
    assert not _mutable_equal(
        Categorical([1, 2], label='a') * 0.75 + Numerical(0, 1, label='b'),
        0.75 * Categorical([1, 2], label='a') + Numerical(0, 1, label='b'),
    )

    assert _mutable_equal(
        MutableList([Categorical([1, 2], label='a'), Categorical([3, 4], label='b')]),
        MutableList([Categorical([1, 2], label='a'), Categorical([3, 4], label='b')]),
    )
    assert not _mutable_equal(
        MutableList([Categorical([1, 2], label='a'), Categorical([3, 4], label='b')]),
        MutableList([Categorical([1, 2], label='a'), Categorical([3, 5], label='b')]),
    )
    assert _mutable_equal(
        MutableDict({'a': Categorical([1, 2], label='a'), 'x': Categorical([3, 4], label='b')}),
        MutableDict({'a': Categorical([1, 2], label='a'), 'x': Categorical([3, 4], label='b')}),
    )
    assert not _mutable_equal(
        MutableDict({'a': Categorical([1, 2], label='a'), 'x': Categorical([3, 4], label='b')}),
        MutableDict({'a': Categorical([1, 2], label='a'), 'x': Categorical([3, 4], label='x')}),
    )

    assert _mutable_equal(np.zeros_like((2, 2)), np.zeros_like((2, 2)))
    assert not _mutable_equal(np.zeros_like((2, 2)), np.ones_like((2, 2)))
