import pytest
from nni.mutable.symbol import Symbol, SymbolicExpression


def test_symbol_repr():
    x, y = Symbol('x'), Symbol('y')
    expr = x * x + y * 2
    assert str(expr) == '(x * x) + (y * 2)'
    assert list(expr.leaf_symbols()) == [x, x, y]
    assert expr.evaluate({'x': 2, 'y': 3}) == 10
    expr = x * x
    assert repr(expr) == f"Symbol('x') * Symbol('x')"


def test_switch_case():
    x, y = Symbol('x'), Symbol('y')
    expr = SymbolicExpression.switch_case(x, {0: y, 1: x * 2})
    assert str(expr) == 'switch_case(x, {0: y, 1: (x * 2)})'
    assert expr.evaluate({'x': 0, 'y': 3}) == 3
    assert expr.evaluate({'x': 1, 'y': 3}) == 2
    with pytest.raises(RuntimeError, match='No matching case'):
        expr.evaluate({'x': 2, 'y': 3})


def test_case():
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    expr = SymbolicExpression.case([(x < y, 17), (x > z, 23), (y > z, 31)])
    assert str(expr) == 'case([((x < y), 17), ((x > z), 23), ((y > z), 31)])'
    assert expr.evaluate({'x': 1, 'y': 2, 'z': 3}) == 17
    assert expr.evaluate({'x': 2, 'y': 1, 'z': 0}) == 23
    assert expr.evaluate({'x': 1, 'y': 2, 'z': 0}) == 17
    with pytest.raises(RuntimeError, match='No matching case'):
        assert expr.evaluate({'x': 2, 'y': 1, 'z': 3})
