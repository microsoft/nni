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
test_evolution_tuner.py
"""

import numpy as np

from unittest import TestCase, main

from nni.evolution_tuner.evolution_tuner import json2space, json2parameter


class EvolutionTunerTestCase(TestCase):
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
        self.assertIn('root[optimizer]-choice', search_space_instance)
        self.assertIn('root[learning_rate]-choice', search_space_instance)

    def test_json2parameter(self):
        """test for json2parameter
        """
        json_search_space = {
            "optimizer":{
                "_type":"choice","_value":["Adam", "SGD"]
            },
            "learning_rate":{
                "_type":"choice",
                "_value":[0.0001, 0.001, 0.002, 0.005, 0.01]
            }
        }
        space = json2space(json_search_space)
        random_state = np.random.RandomState()
        is_rand = dict()
        for item in space:
            is_rand[item] = True
        search_space_instance = json2parameter(json_search_space, is_rand, random_state)
        self.assertIn(search_space_instance["optimizer"]["_index"], range(2))
        self.assertIn(search_space_instance["optimizer"]["_value"], ["Adam", "SGD"])
        self.assertIn(search_space_instance["learning_rate"]["_index"], range(5))
        self.assertIn(search_space_instance["learning_rate"]["_value"], [0.0001, 0.001, 0.002, 0.005, 0.01])


if __name__ == '__main__':
    main()
