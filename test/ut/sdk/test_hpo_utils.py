from nni.common.hpo_utils import validate_search_space

good = {
    'choice': { '_type': 'choice', '_value': ['a', 'b'] },
    'randint': { '_type': 'randint', '_value': [1, 10] },
    'uniform': { '_type': 'uniform', '_value': [0, 1.0] },
    'quniform': { '_type': 'quniform', '_value': [1, 10, 0.1] },
    'loguniform': { '_type': 'loguniform', '_value': [0.001, 0.1] },
    'qloguniform': { '_type': 'qloguniform', '_value': [0.001, 0.1, 0.001] },
    'normal': { '_type': 'normal', '_value': [0, 0.1] },
    'qnormal': { '_type': 'qnormal', '_value': [0.5, 0.1, 0.1] },
    'lognormal': { '_type': 'lognormal', '_value': [0.0, 1] },
    'qlognormal': { '_type': 'qlognormal', '_value': [-1, 1, 0.1] },
}
good_partial = {
    'choice': good['choice'],
    'randint': good['randint'],
}
good_nested = {
    'outer': {
        '_type': 'choice',
        '_value': [
            { '_name': 'empty' },
            { '_name': 'a', 'a_1': { '_type': 'choice', '_value': ['a', 'b'] } }
        ]
    }
}

bad_type = 'x'
bad_spec_type = { 'x': [1, 2, 3] }
bad_fields = { 'x': { 'type': 'choice', 'value': ['a', 'b'] } }
bad_type_name = { 'x': { '_type': 'choic', '_value': ['a'] } }
bad_value = { 'x': { '_type': 'choice', '_value': 'ab' } }
bad_choice_args = { 'x': { '_type': 'choice', '_value': [ 'a', object() ] } }
bad_2_args = { 'x': { '_type': 'randint', '_value': [1, 2, 3] } }
bad_3_args = { 'x': { '_type': 'quniform', '_value': [0] } }
bad_int_args = { 'x': { '_type': 'randint', '_value': [1.0, 2.0] } }
bad_float_args = { 'x': { '_type': 'uniform', '_value': ['0.1', '0.2'] } }
bad_low_high = { 'x': { '_type': 'quniform', '_value': [2, 1, 0.1] } }
bad_log = { 'x': { '_type': 'loguniform', '_value': [0, 1] } }
bad_sigma = { 'x': { '_type': 'normal', '_value': [0, 0] } }

def test_hpo_utils():
    assert validate_search_space(good, raise_exception=False)
    assert validate_search_space(good_nested, raise_exception=False)
    assert not validate_search_space(bad_type, raise_exception=False)
    assert not validate_search_space(bad_spec_type, raise_exception=False)
    assert not validate_search_space(bad_fields, raise_exception=False)
    assert not validate_search_space(bad_type_name, raise_exception=False)
    assert not validate_search_space(bad_value, raise_exception=False)
    assert not validate_search_space(bad_choice_args, raise_exception=False)
    assert not validate_search_space(bad_2_args, raise_exception=False)
    assert not validate_search_space(bad_3_args, raise_exception=False)
    assert not validate_search_space(bad_int_args, raise_exception=False)
    assert not validate_search_space(bad_float_args, raise_exception=False)
    assert not validate_search_space(bad_low_high, raise_exception=False)
    assert not validate_search_space(bad_log, raise_exception=False)
    assert not validate_search_space(bad_sigma, raise_exception=False)

    assert validate_search_space(good_partial, ['choice', 'randint'], False)
    assert not validate_search_space(good, ['choice', 'randint'], False)

if __name__ == '__main__':
    test_hpo_utils()
