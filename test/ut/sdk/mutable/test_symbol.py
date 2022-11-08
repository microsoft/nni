from nni.mutable.symbol import Symbol


def test_symbol_repr():
    x, y = Symbol('x'), Symbol('y')
    expr = x * x + y * 2
    assert str(expr)
    (x * x) + (y * 2)
    list(expr.leaf_symbols())
    [Symbol('x'), Symbol('x'), Symbol('y')]
    expr.evaluate({'x': 2, 'y': 3})
