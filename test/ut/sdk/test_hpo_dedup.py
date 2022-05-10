import numpy as np

import nni
from nni.algorithms.hpo.random_tuner import suggest
from nni.common.hpo_utils import Deduplicator, deformat_parameters, format_search_space

seed = np.random.default_rng().integers(2 ** 31)
print(seed)
rng = np.random.default_rng(seed)

finite_space = {
    'x': {'_type': 'choice', '_value': ['a', 'b']},
    'y': {'_type': 'quniform', '_value': [0, 1, 0.6]},
    'z': {'_type': 'normal', '_value': [1, 0]},
}

infinite_space = {
    'x': {'_type': 'choice', '_value': ['a', 'b']},
    'y': {'_type': 'uniform', '_value': [0, 1]},
}

nested_space = {
    'outer': {
        '_type': 'choice',
        '_value': [
            {'_name': 'A', 'x': {'_type': 'choice', '_value': ['a', 'b']}},
            {'_name': 'B', 'y': {'_type': 'uniform', '_value': [0, 1]}},
        ]
    }
}

def test_dedup_finite():
    space = format_search_space(finite_space)
    dedup = Deduplicator(space)
    params = []
    exhausted = False
    try:
        for i in range(7):
            p = dedup(suggest(rng, space))
            params.append(deformat_parameters(p, space))
    except nni.NoMoreTrialError:
        exhausted = True
    params = sorted(params, key=(lambda p: (p['x'], p['y'], p['z'])))
    assert exhausted
    assert params == [
        {'x': 'a', 'y': 0.0, 'z': 1.0},
        {'x': 'a', 'y': 0.6, 'z': 1.0},
        {'x': 'a', 'y': 1.0, 'z': 1.0},
        {'x': 'b', 'y': 0.0, 'z': 1.0},
        {'x': 'b', 'y': 0.6, 'z': 1.0},
        {'x': 'b', 'y': 1.0, 'z': 1.0},
    ]

def test_dedup_infinite():
    space = format_search_space(infinite_space)
    dedup = Deduplicator(space)
    for i in range(10):
        p = suggest(rng, space)
        assert dedup(p) is p

def test_dedup_nested():
    space = format_search_space(nested_space)
    dedup = Deduplicator(space)
    params = set()
    for i in range(10):
        p = dedup(suggest(rng, space))
        s = nni.dump(deformat_parameters(p, space), sort_keys=True)
        assert s not in params
        params.add(s)

if __name__ == '__main__':
    test_dedup_finite()
    test_dedup_infinite()
    test_dedup_nested()
