# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

os.environ['NNI_PLATFORM'] = 'unittest'

import nni
import nni.runtime.platform.test as test_platform
import nni.trial

from unittest import TestCase, main



class SmartParamTestCase(TestCase):
    def setUp(self):
        params = {
            'test_smartparam/choice1/choice': 'a',
            'test_smartparam/choice2/choice': '3*2+1',
            'test_smartparam/choice3/choice': '[1, 2]',
            'test_smartparam/choice4/choice': '{"a", 2}',
            'test_smartparam/func/function_choice': 'bar',
            'test_smartparam/lambda_func/function_choice': "lambda: 2*3",
            'mutable_block_66':{
                'mutable_layer_0':{
                    'chosen_layer': 'conv2D(size=5)',
                    'chosen_inputs': ['y']
                }
            }
        }
        nni.trial._params = { 'parameter_id': 'test_trial', 'parameters': params }


    def test_specified_name(self):
        val = nni.choice({'a': 'a', '3*2+1': 3*2+1, '[1, 2]': [1, 2], '{"a", 2}': {"a", 2}}, name = 'choice1', key='test_smartparam/choice1/choice')
        self.assertEqual(val, 'a')
        val = nni.choice({'a': 'a', '3*2+1': 3*2+1, '[1, 2]': [1, 2], '{"a", 2}': {"a", 2}}, name = 'choice2', key='test_smartparam/choice2/choice')
        self.assertEqual(val, 7)
        val = nni.choice({'a': 'a', '3*2+1': 3*2+1, '[1, 2]': [1, 2], '{"a", 2}': {"a", 2}}, name = 'choice3', key='test_smartparam/choice3/choice')
        self.assertEqual(val, [1, 2])
        val = nni.choice({'a': 'a', '3*2+1': 3*2+1, '[1, 2]': [1, 2], '{"a", 2}': {"a", 2}}, name = 'choice4', key='test_smartparam/choice4/choice')
        self.assertEqual(val, {"a", 2})

    def test_func(self):
        val = nni.function_choice({'foo': foo, 'bar': bar}, name='func', key='test_smartparam/func/function_choice')
        self.assertEqual(val, 'bar')

    def test_lambda_func(self):
        val = nni.function_choice({"lambda: 2*3": lambda: 2*3, "lambda: 3*4": lambda: 3*4}, name = 'lambda_func', key='test_smartparam/lambda_func/function_choice')
        self.assertEqual(val, 6)

    def test_mutable_layer(self):
        layer_out = nni.mutable_layer('mutable_block_66',
                'mutable_layer_0', {'conv2D(size=3)': conv2D, 'conv2D(size=5)': conv2D}, {'conv2D(size=3)':
                {'size':3}, 'conv2D(size=5)': {'size':5}}, [100], {'x':1,'y':2}, 1, 'classic_mode')
        self.assertEqual(layer_out, [100, 2, 5])
        


def foo():
    return 'foo'

def bar():
    return 'bar'

def conv2D(inputs, size=3):
    return inputs[0] + inputs[1] + [size]

if __name__ == '__main__':
    main()
