from collections import Counter

import nni
from nni.mutable._notimplemented import randint, lognormal, qlognormal

def test_choice():
    t = nni.choice('t', ['a', 'b', 'c'])
    assert repr(t) == "Categorical(['a', 'b', 'c'], label='t')"

def test_randint():
    t = randint('x', 1, 5)
    assert repr(t) == "RandomInteger([1, 2, 3, 4], label='x')"

def test_uniform():
    t = nni.uniform('x', 0, 1)
    assert repr(t) == "Numerical(0, 1, label='x')"

def test_quniform():
    t = nni.quniform('x', 2.5, 5.5, 2.)
    assert repr(t) == "Numerical(2.5, 5.5, q=2.0, label='x')"
    t = nni.quniform('x', 0.5, 3.5, 1).int()
    counter = Counter()
    for _ in range(900):
        counter[t.random()] += 1
    for key, value in counter.items():
        assert 250 <= value <= 350
        assert isinstance(key, int)
        assert key in [1, 2, 3]

def test_loguniform():
    t = nni.loguniform('x', 1e-5, 1e-3)
    assert repr(t) == "Numerical(1e-05, 0.001, log_distributed=True, label='x')"
    for _ in range(100):
        assert 1e-5 < t.random() < 1e-3

def test_qloguniform():
    t = nni.qloguniform('x', 1e-5, 1e-3, 1e-4)
    assert repr(t) == "Numerical(1e-05, 0.001, q=0.0001, log_distributed=True, label='x')"
    for x in t.grid(granularity=8):
        assert (x == 1e-5 or abs(x - round(x / 1e-4) * 1e-4) < 1e-12) and 1e-5 <= x <= 1e-3

def test_normal():
    t = nni.normal('x', 0, 1)
    assert repr(t) == "Numerical(-inf, inf, mu=0, sigma=1, label='x')"
    assert -4 < t.random() < 4

def test_qnormal():
    t = nni.qnormal('x', 0., 1., 0.1)
    assert repr(t) == "Numerical(-inf, inf, mu=0.0, sigma=1.0, q=0.1, label='x')"

def test_lognormal():
    t = lognormal('x', 4., 2.)
    assert repr(t) == "Numerical(-inf, inf, mu=4.0, sigma=2.0, log_distributed=True, label='x')"
    assert 54 < list(t.grid(granularity=1))[0] < 55

def test_qlognormal():
    t = qlognormal('x', 4., 2., 1.)
    assert repr(t) == "Numerical(-inf, inf, mu=4.0, sigma=2.0, q=1.0, log_distributed=True, label='x')"
