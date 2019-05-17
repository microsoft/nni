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

from unittest import TestCase, main

import nni
from nni.utils import split_index


class UtilsTestCase(TestCase):
    def test_split_index_normal(self):
        """test for normal search space
        """
        normal__params_with_index = {
            "dropout_rate": {
                "_index" : 1,
                "_value" : 0.9
            },
            "hidden_size": {
                "_index" : 1,
                "_value" : 512
            }
        }
        normal__params= {
            "dropout_rate": 0.9,
            "hidden_size": 512
        }

        params = split_index(normal__params_with_index)
        self.assertEqual(params, normal__params)

    def test_split_index_nested(self):
        """test for nested search space
        """
        nested_params_with_index = {
            "layer0": {
                "_name": "Avg_pool",
                "pooling_size":{
                    "_index" : 1,
                    "_value" : 2
                } 
            },
            "layer1": {
                "_name": "Empty"
            },
            "layer2": {
                "_name": "Max_pool",
                "pooling_size": {
                    "_index" : 2,
                    "_value" : 3
                } 
            },
            "layer3": {
                "_name": "Conv",
                "kernel_size":  {
                    "_index" : 3,
                    "_value" : 5
                },
                "output_filters":  {
                    "_index" : 3,
                    "_value" : 64
                }
            }
        }
        nested_params =  {
            "layer0": {
                "_name": "Avg_pool",
                "pooling_size": 2
            },
            "layer1": {
                "_name": "Empty"
            },
            "layer2": {
                "_name": "Max_pool",
                "pooling_size": 3
            },
            "layer3": {
                "_name": "Conv",
                "kernel_size": 5,
                "output_filters": 64
            }
        }
        params = split_index(nested_params_with_index)
        self.assertEqual(params, nested_params)


if __name__ == '__main__':
    main()