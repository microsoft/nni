# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import json
import logging
import os
import random
import shutil
import sys
from collections import deque
from unittest import TestCase, main

from nni.tuner import Tuner
from nni.regularized_evolution_tuner.regularized_evolution_tuner import RegularizedEvolutionTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_tuner')


class ClassicNasTestCase(TestCase):
    def setUp(self):
        self.test_round = 3
        self.params_each_round = 50
        self.exhaustive = False

    def check_range(self, generated_params, search_space):
        for params in generated_params:
            for k in params:
                v = params[k]
                items = search_space[k]
                if items['_type'] == 'layer_choice':
                    self.assertIn(v['_value'], items['_value'])
                elif items['_type'] == 'input_choice':
                    for choice in v['_value']:
                        self.assertIn(choice, items['_value']['candidates'])
                else:
                    raise KeyError

    def send_trial_result(self, tuner, parameter_id, parameters, metrics):
        tuner.receive_trial_result(parameter_id, parameters, metrics)
        tuner.trial_end(parameter_id, True)

    def search_space_test_one(self, tuner_factory, search_space):
        tuner = tuner_factory()
        self.assertIsInstance(tuner, Tuner)
        tuner.update_search_space(search_space)
        for i in range(self.test_round):
            queue = deque()
            parameters = tuner.generate_multiple_parameters(list(range(i * self.params_each_round,
                                                                       (i + 1) * self.params_each_round)),
                                                            st_callback=self.send_trial_callback(queue))
            logger.debug(parameters)
            self.check_range(parameters, search_space)
            for k in range(min(len(parameters), self.params_each_round)):
                self.send_trial_result(tuner, self.params_each_round * i + k, parameters[k], random.uniform(-100, 100))
            while queue:
                id_, params = queue.popleft()
                self.check_range([params], search_space)
                self.send_trial_result(tuner, id_, params, random.uniform(-100, 100))
            if not parameters and not self.exhaustive:
                raise ValueError("No parameters generated")

    def send_trial_callback(self, param_queue):
        def receive(*args):
            param_queue.append(tuple(args))
        return receive

    def search_space_test_all(self, tuner_factory):
        # Since classic tuner should support only LayerChoice and InputChoice,
        # ignore type and fail type are dismissed here. 
        with open(os.path.join(os.path.dirname(__file__), "assets/classic_nas_search_space.json"), "r") as fp:
            search_space_all = json.load(fp)
        full_supported_search_space = dict()
        for single in search_space_all:
            space = search_space_all[single]
            single_search_space = {single: space}
            self.search_space_test_one(tuner_factory, single_search_space)
            full_supported_search_space.update(single_search_space)
        logger.info("Full supported search space: %s", full_supported_search_space)
        self.search_space_test_one(tuner_factory, full_supported_search_space)

    def test_evo_nas_tuner(self):
        tuner_fn = lambda: EvoNasTuner()
        self.search_space_test_all(tuner_fn)


if __name__ == '__main__':
    main()

