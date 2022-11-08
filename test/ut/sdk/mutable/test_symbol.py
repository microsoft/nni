from nni.mutable.symbol import Symbol


def test_symbol_repr():
    x, y = Symbol('x'), Symbol('y')
    expr = x * x + y * 2
    assert str(expr) == '(x * x) + (y * 2)'
    assert list(expr.leaf_symbols()) == [x, x, y]
    assert expr.evaluate({'x': 2, 'y': 3}) == 10
    expr = x * x
    assert repr(expr) == f"Symbol('x') * Symbol('x')"
