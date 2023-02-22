import pytest

from nni.mutable import *


def _frozen_context_middle():
    assert frozen_context.current() == {'a': 1, 'b': 2}

    with frozen_context.bypass():
        assert frozen_context.current() is None


def test_frozen_context():
    with frozen_context({'a': 1, 'b': 2}):
        _frozen_context_middle()

    assert frozen_context.current() is None

    with frozen_context({'a': 1, 'b': 2}):
        with frozen_context({'c': 3}):
            assert frozen_context.current() == {'a': 1, 'b': 2, 'c': 3}
            with frozen_context.bypass():
                assert frozen_context.current() == {'a': 1, 'b': 2}


def _frozen_context_complex_middle():
    assert frozen_context.current() == {}

    frozen_context.update({'a': 1, 'b': 2})
    assert frozen_context.current() == {'a': 1, 'b': 2}
    frozen_context.update({'c': 3})
    assert frozen_context.current() == {'a': 1, 'b': 2, 'c': 3}

    with frozen_context({'c': 4}):
        assert frozen_context.current() == {'a': 1, 'b': 2, 'c': 4}
        frozen_context.update({'d': 5})
        assert frozen_context.current() == {'a': 1, 'b': 2, 'c': 4, 'd': 5}
    assert frozen_context.current() == {'a': 1, 'b': 2, 'c': 3}


def test_frozen_context_complex():
    assert frozen_context.current() is None
    with frozen_context():
        _frozen_context_complex_middle()
    assert frozen_context.current() is None

    with frozen_context('anything'):
        with pytest.raises(TypeError, match='dict'):
            frozen_context.current()


def test_ensure_frozen(caplog):
    assert ensure_frozen(Categorical([1, 2, 3]), strict=False) == 1
    assert ensure_frozen(Categorical([1, 2, 3], label='a'), sample={'a': 2}, strict=False) == 2
    assert ensure_frozen('anything', strict=False) == 'anything'

    with pytest.raises(RuntimeError, match='context'):
        ensure_frozen(Categorical([1, 2, 3], label='a'))

    with frozen_context({'a': 1, 'b': 2}):
        assert ensure_frozen(Categorical([1, 2], label='a')) == 1
        assert ensure_frozen(Categorical([1, 2], label='b')) == 2
        assert ensure_frozen(Categorical([1, 2], label='a') + Categorical([1, 2], label='b')) == 3
        with pytest.raises(SampleValidationError, match='missing from'):
            ensure_frozen(Categorical([1, 2], label='c'))
        assert 'add_mutable' in caplog.text

        with frozen_context.bypass():
            assert ensure_frozen(Categorical([1, 2], label='a', default=2), strict=False) == 2
            with pytest.raises(RuntimeError, match='context'):
                ensure_frozen(Categorical([1, 2], label='a'), retries=-1)


def _func_with_ensure_frozen(a, b):
    return ensure_frozen(a + b)


def test_frozen_factory():
    func = frozen_factory(_func_with_ensure_frozen, {'a': 1, 'b': 2})
    assert func(Numerical(0, 2, label='a'), Numerical(0, 2, label='b')) == 3
