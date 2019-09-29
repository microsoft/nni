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
import copy
import glob
import json
import logging
import os
import shutil
from unittest import TestCase, main

from nni.batch_tuner import BatchTuner
from nni.evolution_tuner import EvolutionTuner
from nni.gp_tuner import GPTuner
from nni.gridsearch_tuner import GridSearchTuner
from nni.hyperopt_tuner import HyperoptTuner
from nni.metis_tuner import MetisTuner
from nni.smac_tuner import SMACTuner
from nni.tuner import Tuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_tuner')


class TunerTestCase(TestCase):
    """
    Targeted at testing functions of built-in tuners, including
        - [ ] load_checkpoint
        - [ ] save_checkpoint
        - [X] update_search_space
        - [X] generate_multiple_parameters
        - [ ] import_data
        - [ ] trial_end
        - [ ] receive_trial_result
    """

    def search_space_test_one(self, tuner_factory, search_space):
        tuner = tuner_factory()
        self.assertIsInstance(tuner, Tuner)
        tuner.update_search_space(search_space)

        parameters = tuner.generate_multiple_parameters(list(range(0, 50)))
        logger.info(parameters)
        if not parameters:  # TODO: not strict
            raise ValueError("No parameters generated")
        return parameters

    def check_range(self, generated_params, search_space):
        EPS = 1E-6
        for param in generated_params:
            for k, v in param.items():
                item = search_space[k]
                if item["_type"] == "choice":
                    self.assertIn(v, item["_value"])
                if item["_type"] == "randint":
                    self.assertIsInstance(v, int)
                    self.assertIn(v, item["_value"])
                if item["_type"] == "uniform":
                    self.assertIsInstance(v, float)
                if item["_type"] in ("randint", "uniform", "quniform", "loguniform", "qloguniform"):
                    self.assertGreaterEqual(v, item["_value"][0])
                    self.assertLessEqual(v, item["_value"][1])
                if item["_type"].startswith("q"):
                    multiple = v / item["_value"][2]
                    if item["_value"][0] + EPS < v < item["_value"][1] - EPS:
                        self.assertAlmostEqual(int(round(multiple) + EPS), multiple)
                if item["_type"] in ("qlognormal", "lognormal"):
                    self.assertGreater(v, 0)

    def search_space_test_all(self, tuner_factory, supported_types=None):
        with open(os.path.join(os.path.dirname(__file__), "assets/search_space.json"), "r") as fp:
            search_space_all = json.load(fp)
        if supported_types is None:
            supported_types = ["choice", "randint", "uniform", "quniform", "loguniform", "qloguniform",
                               "normal", "qnormal", "lognormal", "qlognormal"]
        full_supported_search_space = dict()
        for single in search_space_all:
            single_keyword = single.split("_")
            space = search_space_all[single]
            expected_fail = not any([t in single_keyword for t in supported_types]) or "fail" in single_keyword
            if "fail" in space:
                if self._testMethodName.split("_", 1)[1] in space.pop("fail"):
                    expected_fail = True
            single_search_space = {single: space}
            print(single, expected_fail)
            if not expected_fail:
                # supports this key
                self.search_space_test_one(tuner_factory, single_search_space)
                full_supported_search_space.update(single_search_space)
            else:
                # unsupported key
                with self.assertRaises(Exception) as cm:
                    self.search_space_test_one(tuner_factory, single_search_space)
                logger.info("{}, {}, {}".format(tuner_factory, single, cm.exception))
        if "batch" not in self._testMethodName:
            logger.info("Full supported search space: {}".format(full_supported_search_space))
            self.search_space_test_one(tuner_factory, full_supported_search_space)

    def test_grid_search(self):
        self.search_space_test_all(lambda: GridSearchTuner(),
                                   supported_types=["choice", "randint", "quniform"])

    def test_tpe(self):
        self.search_space_test_all(lambda: HyperoptTuner("tpe"))

    def test_random_search(self):
        self.search_space_test_all(lambda: HyperoptTuner("random_search"))

    def test_anneal(self):
        self.search_space_test_all(lambda: HyperoptTuner("anneal"))

    def test_smac(self):
        self.search_space_test_all(lambda: SMACTuner(),
                                   supported_types=["choice", "randint", "uniform", "quniform", "loguniform"])

    def test_batch(self):
        self.search_space_test_all(lambda: BatchTuner(),
                                   supported_types=["choice"])

    def test_evolution(self):
        # Needs enough population size, otherwise it will throw a runtime error
        self.search_space_test_all(lambda: EvolutionTuner(population_size=100))

    def test_gp(self):
        self.search_space_test_all(lambda: GPTuner())

    def test_metis(self):
        self.search_space_test_all(lambda: MetisTuner(),
                                   supported_types=["choice", "randint", "uniform", "quniform"])

    def test_networkmorphism(self):
        pass

    def test_ppo(self):
        pass

    def tearDown(self):
        file_list = glob.glob("smac3*") + ["param_config_space.pcs", "scenario.txt", "model_path"]
        for file in file_list:
            if os.path.exists(file):
                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)


if __name__ == '__main__':
    main()
