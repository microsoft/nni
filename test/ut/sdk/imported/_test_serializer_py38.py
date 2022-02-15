import nni


def test_positional_only():
    def foo(a, b, /, c):
        pass

    d = nni.trace(foo)(1, 2, c=3)
    assert d.trace_args == [1, 2]
    assert d.trace_kwargs == dict(c=3)
