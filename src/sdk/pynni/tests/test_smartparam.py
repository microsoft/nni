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


import nni
import nni.platform.test as test_platform
import nni.trial

from unittest import TestCase, main


lineno1 = 48
lineno2 = 58

class SmartParamTestCase(TestCase):
    def setUp(self):
        params = {
            'test_smartparam/choice1/choice': 2,
            'test_smartparam/#{:d}/uniform'.format(lineno1): '5',
            'test_smartparam/func/function_choice': 1,
            'test_smartparam/#{:d}/function_choice'.format(lineno2): 0
        }
        nni.trial._params = { 'parameter_id': 'test_trial', 'parameters': params }


    def test_specified_name(self):
        val = nni.choice('a', 'b', 'c', name = 'choice1')
        self.assertEqual(val, 'c')

    def test_default_name(self):
        val = nni.uniform(1, 10)  # NOTE: assign this line number to lineno1
        self.assertEqual(val, '5')

    def test_specified_name_func(self):
        val = nni.function_choice(foo, bar, name = 'func')
        self.assertEqual(val, 'bar')

    def test_default_name_func(self):
        val = nni.function_choice(
            lambda: max(1, 2, 3),
            lambda: 2 * 2  # NOTE: assign this line number to lineno2
        )
        self.assertEqual(val, 3)


def foo():
    return 'foo'

def bar():
    return 'bar'


if __name__ == '__main__':
    main()
