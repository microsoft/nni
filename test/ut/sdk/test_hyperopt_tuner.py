# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
test_hyperopt_tuner.py
"""

from unittest import TestCase, main

import hyperopt as hp

from nni.algorithms.hpo.hyperopt_tuner.hyperopt_tuner import json2space, json2parameter, json2vals, HyperoptTuner


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

    def test_tuner_generate(self):
        for algorithm in ["tpe", "random_search", "anneal"]:
            tuner = HyperoptTuner(algorithm)
            choice_list = ["a", "b", 1, 2]
            tuner.update_search_space({
                "a": {
                    "_type": "randint",
                    "_value": [1, 3]
                },
                "b": {
                    "_type": "choice",
                    "_value": choice_list
                }
            })
            for k in range(30):
                # sample multiple times
                param = tuner.generate_parameters(k)
                print(param)
                self.assertIsInstance(param["a"], int)
                self.assertGreaterEqual(param["a"], 1)
                self.assertLessEqual(param["a"], 2)
                self.assertIn(param["b"], choice_list)


if __name__ == '__main__':
    main()
