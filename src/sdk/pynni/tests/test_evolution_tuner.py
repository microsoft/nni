# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
test_evolution_tuner.py
"""

import numpy as np

from unittest import TestCase, main

from nni.utils import json2space, json2parameter


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
