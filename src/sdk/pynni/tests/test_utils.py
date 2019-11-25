# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
