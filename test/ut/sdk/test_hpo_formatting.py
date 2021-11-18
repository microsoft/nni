from math import exp, log
from nni.common.hpo_utils import deformat_parameters, format_search_space

user_space = {
    'pool': { '_type': 'choice', '_value': ['max', 'min', 'avg'] },
    'kernel': { '_type': 'randint', '_value': [2, 8] },
    'D': {  # distribution
        '_type': 'choice',
        '_value': [
            {
                '_name': 'UNIFORM',
                'dropout': { '_type': 'uniform', '_value': [0.5, 0.9] },
                'hidden': { '_type': 'quniform', '_value': [100, 1000, 3] },
                'U_lr': { '_type': 'loguniform', '_value': [0.0001, 0.1] },
                'U_batch': { '_type': 'qloguniform', '_value': [16.0, 128.0, 0.725] },
            },
            {
                '_name': 'NORMAL',
                'dropout': { '_type': 'normal', '_value': [0.7, 0.2] },
                'hidden': { '_type': 'qnormal', '_value': [500, 200, 3] },
                'N_lr': { '_type': 'lognormal', '_value': [-6, 3] },
                'N_batch': { '_type': 'qlognormal', '_value': [3.5, 1.2, 0.725] },
            },
            {
                '_name': 'EMPTY',
            },
        ]
    },
    'not_nested': {
        '_type': 'choice',
        '_value': [
            {'x': 0, 'y': 0},
            {'x': 1, 'y': 2},
        ],
    },
}


spec_names = ['pool', 'kernel', 'D', 'dropout', 'hidden', 'U_lr', 'U_batch', 'dropout', 'hidden', 'N_lr', 'N_batch', 'not_nested']
spec_types = ['choice', 'randint', 'choice', 'uniform', 'quniform', 'loguniform', 'qloguniform', 'normal', 'qnormal', 'lognormal', 'qlognormal', 'choice']
spec_values = [['max','min','avg'], [2,8], user_space['D']['_value'], [0.5,0.9], [100.0,1000.0,3.0], [0.0001,0.1], [16.0,128.0,0.725], [0.7,0.2], [500.0,200.0,3.0], [-6.0,3.0], [3.5,1.2,0.725], [{'x':0,'y':0},{'x':1,'y':2}]]
spec_keys = [('pool',), ('kernel',), ('D',), ('D',0,'dropout'), ('D',0,'hidden'), ('D',0,'U_lr'), ('D',0,'U_batch'), ('D',1,'dropout'), ('D',1,'hidden'), ('D',1,'N_lr'), ('D',1,'N_batch'), ('not_nested',)]
spec_categoricals = [True, True, True, False, False, False, False, False, False, False, False, True]
spec_sizes = [3, 6, 3, None, None, None, None, None, None, None, None, 2]
spec_lows = [None, None, None, 0.5, 100.0, log(0.0001), log(16.0), None, None, None, None, None]
spec_highs = [None, None, None, 0.9, 1000.0, log(0.1), log(128.0), None, None, None, None, None]
spec_normals = [None, None, None, False, False, False, False, True, True, True, True, None]
spec_mus = [None, None, None, None, None, None, None, 0.7, 500.0, -6.0, 3.5, None]
spec_sigmas = [None, None, None, None, None, None, None, 0.2, 200.0, 3.0, 1.2, None]
spec_qs = [None, None, None, None, 3.0, None, 0.725, None, 3.0, None, 0.725, None]
spec_clips = [None, None, None, None, (100.0,1000.0), None, (16.0,128.0), None, None, None, None, None]
spec_logs = [None, None, None, False, False, True, True, False, False, True, True, None]

def test_formatting():
    internal_space = format_search_space(user_space)
    assert all(key == value.key for key, value in internal_space.items())
    specs = list(internal_space.values())
    assert spec_names == [spec.name for spec in specs]
    assert spec_types == [spec.type for spec in specs]
    assert spec_values == [spec.values for spec in specs]
    assert spec_keys == [spec.key for spec in specs]
    assert spec_categoricals == [spec.categorical for spec in specs]
    assert spec_sizes == [spec.size for spec in specs]
    assert spec_lows == [spec.low for spec in specs]
    assert spec_highs == [spec.high for spec in specs]
    assert spec_normals == [spec.normal_distributed for spec in specs]
    assert spec_mus == [spec.mu for spec in specs]
    assert spec_sigmas == [spec.sigma for spec in specs]
    assert spec_qs == [spec.q for spec in specs]
    assert spec_clips == [spec.clip for spec in specs]
    assert spec_logs == [spec.log_distributed for spec in specs]


internal_params_1 = {
    ('pool',): 0,
    ('kernel',): 5,
    ('D',): 0,
    ('D',0,'dropout'): 0.7,
    ('D',0,'hidden'): 100.1,  # round to 99.0, then clip to 100.0
    ('D',0,'U_lr'): -4.6,
    ('D',0,'U_batch'): 4.0,
    ('not_nested',): 0,
}

user_params_1 = {
    'pool': 'max',
    'kernel': 7,
    'D': {
        '_name': 'UNIFORM',
        'dropout': 0.7,
        'hidden': 100.0,
        'U_lr': exp(-4.6),
        'U_batch': 54.375,
    },
    'not_nested': {'x': 0, 'y': 0},
}

internal_params_2 = {
    ('pool',): 2,
    ('kernel',): 0,
    ('D',): 1,
    ('D',1,'dropout'): 0.7,
    ('D',1,'hidden'): 100.1,
    ('D',1,'N_lr'): -4.6,
    ('D',1,'N_batch'): 4.0,
    ('not_nested',): 1,
}

user_params_2 = {
    'pool': 'avg',
    'kernel': 2,
    'D': {
        '_name': 'NORMAL',
        'dropout': 0.7,
        'hidden': 99.0,
        'N_lr': exp(-4.6),
        'N_batch': 54.375,
    },
    'not_nested': {'x': 1, 'y': 2},
}

internal_params_3 = {
    ('pool',): 1,
    ('kernel',): 1,
    ('D',): 2,
    ('not_nested',): 1,
}

user_params_3 = {
    'pool': 'min',
    'kernel': 3,
    'D': {
        '_name': 'EMPTY',
    },
    'not_nested': {'x': 1, 'y': 2},
}

def test_deformatting():
    internal_space = format_search_space(user_space)
    assert deformat_parameters(internal_params_1, internal_space) == user_params_1
    assert deformat_parameters(internal_params_2, internal_space) == user_params_2
    assert deformat_parameters(internal_params_3, internal_space) == user_params_3


def test_activate():
    internal_space = format_search_space(user_space)

    assert internal_space[('pool',)].is_activated_in({})

    partial = { ('pool',): 1, ('kernel',): 1, ('D',): 0 }
    assert internal_space[('D', 0, 'dropout')].is_activated_in(partial)
    assert internal_space[('D', 0, 'U_lr')].is_activated_in(partial)
    assert not internal_space[('D', 1, 'dropout')].is_activated_in(partial)
    assert not internal_space[('D', 1, 'N_lr')].is_activated_in(partial)

    partial = { ('pool',): 1, ('kernel',): 1, ('D',): 2 }
    assert not internal_space[('D', 0, 'dropout')].is_activated_in(partial)
    assert not internal_space[('D', 0, 'U_lr')].is_activated_in(partial)
    assert not internal_space[('D', 1, 'dropout')].is_activated_in(partial)
    assert not internal_space[('D', 1, 'N_lr')].is_activated_in(partial)

    assert internal_space[('not_nested',)].is_activated_in(partial)


if __name__ == '__main__':
    test_formatting()
    test_deformatting()
    test_activate()
