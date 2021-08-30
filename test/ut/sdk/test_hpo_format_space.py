from nni.common.hpo_utils import format_search_space, deformat_parameters

user_space = {
    'dropout_rate': { '_type': 'uniform', '_value': [0.5, 0.9] },
    'conv_size': { '_type': 'choice', '_value': [2, 3, 5, 7] },
    'hidden_size': { '_type': 'qloguniform', '_value': [128, 1024, 1] },
    'batch_size': { '_type': 'randint', '_value': [16, 32] },
    'learning_rate': { '_type': 'loguniform', '_value': [0.0001, 0.1] },
    'nested': {
        '_type': 'choice',
        '_value': [
            {
                '_name': 'empty',
            },
            {
                '_name': 'double_nested',
                'xy': {
                    '_type': 'choice',
                    '_value': [
                        {
                            '_name': 'x',
                            'x': { '_type': 'normal', '_value': [0, 1.0] },
                        },
                        {
                            '_name': 'y',
                            'y': { '_type': 'qnormal', '_value': [0, 1, 0.5] },
                        },
                    ],
                },
                'z': { '_type': 'quniform', '_value': [-0.5, 0.5, 0.1] },
            },
            {
                '_name': 'common',
                'x': { '_type': 'lognormal', '_value': [1, 0.1] },
                'y': { '_type': 'qlognormal', '_value': [-1, 1, 0.1] },
            },
        ],
    },
}

internal_space_simple = [  # the full internal space is too long, omit None and False values here
    {'name':'dropout_rate', 'type':'uniform', 'values':[0.5,0.9], 'key':('dropout_rate',), 'low':0.5, 'high':0.9},
    {'name':'conv_size', 'type':'choice', 'values':[2,3,5,7], 'key':('conv_size',), 'categorical':True, 'size':4},
    {'name':'hidden_size', 'type':'qloguniform', 'values':[128,1024,1], 'key':('hidden_size',), 'low':128.0, 'high':1024.0, 'q':1.0, 'log_distributed':True},
    {'name':'batch_size', 'type':'randint', 'values':[16,32], 'key':('batch_size',), 'categorical':True, 'size':16},
    {'name':'learning_rate', 'type':'loguniform', 'values':[0.0001,0.1], 'key':('learning_rate',), 'low':0.0001, 'high':0.1, 'log_distributed':True},
    {'name':'nested', 'type':'choice', '_value_names':['empty','double_nested','common'], 'key':('nested',), 'categorical':True, 'size':3, 'nested_choice':True},
    {'name':'xy', 'type':'choice', '_value_names':['x','y'], 'key':('nested','xy'), 'parent_index':1, 'categorical':True, 'size':2, 'nested_choice':True},
    {'name':'x', 'type':'normal', 'values':[0,1.0], 'key':('nested','xy','x'), 'parent_index':0, 'normal_distributed':True, 'mu':0.0, 'sigma':1.0},
    {'name':'y', 'type':'qnormal', 'values':[0,1,0.5], 'key':('nested','xy','y'), 'parent_index':1, 'normal_distributed':True, 'mu':0.0, 'sigma':1.0, 'q':0.5},
    {'name':'z', 'type':'quniform', 'values':[-0.5,0.5,0.1], 'key':('nested','z'), 'parent_index':1, 'low':-0.5, 'high':0.5, 'q':0.1},
    {'name':'x', 'type':'lognormal', 'values':[1,0.1], 'key':('nested','x'), 'parent_index':2, 'normal_distributed':True, 'mu':1.0, 'sigma':0.1, 'log_distributed':True},
    {'name':'y', 'type':'qlognormal', 'values':[-1,1,0.1], 'key':('nested','y'), 'parent_index':2, 'normal_distributed':True, 'mu':-1.0, 'sigma':1.0, 'q':0.1, 'log_distributed':True},
]

def test_format_search_space():
    formatted = format_search_space(user_space)
    for spec, expected in zip(formatted.values(), internal_space_simple):
        for key, value in spec._asdict().items():
            if key == 'values' and '_value_names' in expected:
                assert [v['_name'] for v in value] == expected['_value_names']
            elif key in expected:
                assert value == expected[key]
            else:
                assert value is None or value == False

internal_parameters = {
    ('dropout_rate',): 0.7,
    ('conv_size',): 2,
    ('hidden_size',): 200.0,
    ('batch_size',): 3,
    ('learning_rate',): 0.0345,
    ('nested',): 1,
    ('nested', 'xy'): 0,
    ('nested', 'xy', 'x'): 0.123,
}

user_parameters = {
    'dropout_rate': 0.7,
    'conv_size': 5,
    'hidden_size': 200.0,
    'batch_size': 19,
    'learning_rate': 0.0345,
    'nested': {
        '_name': 'double_nested',
        'xy': {
            '_name': 'x',
            'x': 0.123,
        },
    },
}

def test_deformat_parameters():
    space = format_search_space(user_space)
    generated = deformat_parameters(internal_parameters, space)
    assert generated == user_parameters

if __name__ == '__main__':
    test_format_search_space()
    test_deformat_parameters()
