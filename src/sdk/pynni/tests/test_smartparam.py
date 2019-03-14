# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import os

os.environ['NNI_PLATFORM'] = 'unittest'

import nni
import nni.platform.test as test_platform
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
            'test_smartparam/lambda_func/function_choice': "lambda: 2*3"
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


def foo():
    return 'foo'

def bar():
    return 'bar'


if __name__ == '__main__':
    main()
