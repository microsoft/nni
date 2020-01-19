# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import json
import logging
import os
import random
import shutil
import sys
from unittest import TestCase, main

from nni.batch_tuner.batch_tuner import BatchTuner
from nni.evolution_tuner.evolution_tuner import EvolutionTuner
from nni.gp_tuner.gp_tuner import GPTuner
from nni.gridsearch_tuner.gridsearch_tuner import GridSearchTuner
from nni.hyperopt_tuner.hyperopt_tuner import HyperoptTuner
from nni.metis_tuner.metis_tuner import MetisTuner

try:
    from nni.smac_tuner.smac_tuner import SMACTuner
except ImportError:
    assert sys.platform == "win32"
from nni.tuner import Tuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_tuner')


class BuiltinTunersTestCase(TestCase):
    """
    Targeted at testing functions of built-in tuners, including
        - [ ] load_checkpoint
        - [ ] save_checkpoint
        - [X] update_search_space
        - [X] generate_multiple_parameters
        - [ ] import_data
        - [ ] trial_end
        - [x] receive_trial_result
    """

    def setUp(self):
        self.test_round = 3
        self.params_each_round = 50
        self.exhaustive = False

    def search_space_test_one(self, tuner_factory, search_space):
        tuner = tuner_factory()
        self.assertIsInstance(tuner, Tuner)
        tuner.update_search_space(search_space)

        for i in range(self.test_round):
            parameters = tuner.generate_multiple_parameters(list(range(i * self.params_each_round,
                                                                       (i + 1) * self.params_each_round)))
            logger.debug(parameters)
            self.check_range(parameters, search_space)
            for k in range(min(len(parameters), self.params_each_round)):
                tuner.receive_trial_result(self.params_each_round * i + k, parameters[k], random.uniform(-100, 100))
            if not parameters and not self.exhaustive:
                raise ValueError("No parameters generated")

    def check_range(self, generated_params, search_space):
        EPS = 1E-6
        for param in generated_params:
            if self._testMethodName == "test_batch":
                param = {list(search_space.keys())[0]: param}
            for k, v in param.items():
                if k.startswith("_mutable_layer"):
                    _, block, layer, choice = k.split("/")
                    cand = search_space[block]["_value"][layer].get(choice)
                    # cand could be None, e.g., optional_inputs_chosen_state
                    if choice == "layer_choice":
                        self.assertIn(v, cand)
                    if choice == "optional_input_size":
                        if isinstance(cand, int):
                            self.assertEqual(v, cand)
                        else:
                            self.assertGreaterEqual(v, cand[0])
                            self.assertLessEqual(v, cand[1])
                    if choice == "optional_inputs":
                        pass  # ignore for now
                    continue
                item = search_space[k]
                if item["_type"] == "choice":
                    self.assertIn(v, item["_value"])
                if item["_type"] == "randint":
                    self.assertIsInstance(v, int)
                if item["_type"] == "uniform":
                    self.assertIsInstance(v, float)
                if item["_type"] in ("randint", "uniform", "quniform", "loguniform", "qloguniform"):
                    self.assertGreaterEqual(v, item["_value"][0])
                    self.assertLessEqual(v, item["_value"][1])
                if item["_type"].startswith("q"):
                    multiple = v / item["_value"][2]
                    print(k, v, multiple, item)
                    if item["_value"][0] + EPS < v < item["_value"][1] - EPS:
                        self.assertAlmostEqual(int(round(multiple)), multiple)
                if item["_type"] in ("qlognormal", "lognormal"):
                    self.assertGreaterEqual(v, 0)
                if item["_type"] == "mutable_layer":
                    for layer_name in item["_value"].keys():
                        self.assertIn(v[layer_name]["chosen_layer"], item["layer_choice"])

    def search_space_test_all(self, tuner_factory, supported_types=None, ignore_types=None, fail_types=None):
        # Three types: 1. supported; 2. ignore; 3. fail.
        # NOTE(yuge): ignore types
        # Supported types are listed in the table. They are meant to be supported and should be correct.
        # Other than those, all the rest are "unsupported", which are expected to produce ridiculous results
        # or throw some exceptions. However, there are certain types I can't check. For example, generate
        # "normal" using GP Tuner returns successfully and results are fine if we check the range (-inf to +inf),
        # but they make no sense: it's not a normal distribution. So they are ignored in tests for now.
        with open(os.path.join(os.path.dirname(__file__), "assets/search_space.json"), "r") as fp:
            search_space_all = json.load(fp)
        if supported_types is None:
            supported_types = ["choice", "randint", "uniform", "quniform", "loguniform", "qloguniform",
                               "normal", "qnormal", "lognormal", "qlognormal"]
        if fail_types is None:
            fail_types = []
        if ignore_types is None:
            ignore_types = []
        full_supported_search_space = dict()
        for single in search_space_all:
            space = search_space_all[single]
            if any(single.startswith(t) for t in ignore_types):
                continue
            expected_fail = not any(single.startswith(t) for t in supported_types) or \
                            any(single.startswith(t) for t in fail_types) or \
                            "fail" in single  # name contains fail (fail on all)
            single_search_space = {single: space}
            if not expected_fail:
                # supports this key
                self.search_space_test_one(tuner_factory, single_search_space)
                full_supported_search_space.update(single_search_space)
            else:
                # unsupported key
                with self.assertRaises(Exception, msg="Testing {}".format(single)) as cm:
                    self.search_space_test_one(tuner_factory, single_search_space)
                logger.info("%s %s %s", tuner_factory, single, cm.exception)
        if not any(t in self._testMethodName for t in ["batch", "grid_search"]):
            # grid search fails for too many combinations
            logger.info("Full supported search space: %s", full_supported_search_space)
            self.search_space_test_one(tuner_factory, full_supported_search_space)

    def test_grid_search(self):
        self.exhaustive = True
        self.search_space_test_all(lambda: GridSearchTuner(),
                                   supported_types=["choice", "randint", "quniform"])

    def test_tpe(self):
        self.search_space_test_all(lambda: HyperoptTuner("tpe"),
                                   ignore_types=["uniform_equal", "qloguniform_equal", "loguniform_equal", "quniform_clip_2"])
        # NOTE: types are ignored because `tpe.py line 465, in adaptive_parzen_normal assert prior_sigma > 0`

    def test_random_search(self):
        self.search_space_test_all(lambda: HyperoptTuner("random_search"))

    def test_anneal(self):
        self.search_space_test_all(lambda: HyperoptTuner("anneal"))

    def test_smac(self):
        if sys.platform == "win32":
            return  # smac doesn't work on windows
        self.search_space_test_all(lambda: SMACTuner(),
                                   supported_types=["choice", "randint", "uniform", "quniform", "loguniform"])

    def test_batch(self):
        self.exhaustive = True
        self.search_space_test_all(lambda: BatchTuner(),
                                   supported_types=["choice"])

    def test_evolution(self):
        # Needs enough population size, otherwise it will throw a runtime error
        self.search_space_test_all(lambda: EvolutionTuner(population_size=100))

    def test_gp(self):
        self.test_round = 1  # NOTE: GP tuner got hanged for multiple testing round
        self.search_space_test_all(lambda: GPTuner(),
                                   supported_types=["choice", "randint", "uniform", "quniform", "loguniform",
                                                    "qloguniform"],
                                   ignore_types=["normal", "lognormal", "qnormal", "qlognormal"],
                                   fail_types=["choice_str", "choice_mixed"])

    def test_metis(self):
        self.test_round = 1  # NOTE: Metis tuner got hanged for multiple testing round
        self.search_space_test_all(lambda: MetisTuner(),
                                   supported_types=["choice", "randint", "uniform", "quniform"],
                                   fail_types=["choice_str", "choice_mixed"])

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
