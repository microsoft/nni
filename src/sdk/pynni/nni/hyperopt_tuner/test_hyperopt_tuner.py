# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
test_hyperopt_tuner.py
"""

from unittest import TestCase, main

import hyperopt as hp

from nni.hyperopt_tuner.hyperopt_tuner import json2space, json2parameter, json2vals


class HyperoptTunerTestCase(TestCase):
    def test_json2space(self):
        """test for json2space
        """
        json_search_space = {
            "optimizer": {
                "_type": "choice",
                "_value": ["Adam", "SGD"]
            },
            "learning_rate": {
                "_type": "choice",
                "_value": [0.0001, 0.001, 0.002, 0.005, 0.01]
            }
        }
        search_space_instance = json2space(json_search_space)
        self.assertIsInstance(search_space_instance["optimizer"],
                              hp.pyll.base.Apply)
        self.assertIsInstance(search_space_instance["learning_rate"],
                              hp.pyll.base.Apply)

    def test_json2parameter(self):
        """test for json2parameter
        """
        json_search_space = {
            "optimizer": {
                "_type": "choice",
                "_value": ["Adam", "SGD"]
            },
            "learning_rate": {
                "_type": "choice",
                "_value": [0.0001, 0.001, 0.002, 0.005, 0.01]
            }
        }
        parameter = {
            'root[learning_rate]-choice': 2,
            'root[optimizer]-choice': 0
        }
        search_space_instance = json2parameter(json_search_space, parameter)
        self.assertEqual(search_space_instance["optimizer"]["_index"], 0)
        self.assertEqual(search_space_instance["optimizer"]["_value"], "Adam")
        self.assertEqual(search_space_instance["learning_rate"]["_index"], 2)
        self.assertEqual(search_space_instance["learning_rate"]["_value"], 0.002)

    def test_json2vals(self):
        """test for json2vals
        """
        json_search_space = {
            "optimizer": {
                "_type": "choice",
                "_value": ["Adam", "SGD"]
            },
            "learning_rate": {
                "_type": "choice",
                "_value": [0.0001, 0.001, 0.002, 0.005, 0.01]
            }
        }
        out_y = dict()
        vals = {
            'optimizer': {
                '_index': 0,
                '_value': 'Adam'
            },
            'learning_rate': {
                '_index': 1,
                '_value': 0.001
            }
        }
        json2vals(json_search_space, vals, out_y)
        self.assertEqual(out_y["root[optimizer]-choice"], 0)
        self.assertEqual(out_y["root[learning_rate]-choice"], 1)


if __name__ == '__main__':
    main()
