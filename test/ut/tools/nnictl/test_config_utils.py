# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest import TestCase, main
from nni.tools.nnictl.config_utils import Config, Experiments

HOME_PATH = str(Path(__file__).parent / "mock/nnictl_metadata")

class CommonUtilsTestCase(TestCase):

    def test_update_experiment(self):
        experiment = Experiments(HOME_PATH)
        experiment.add_experiment('xOpEwA5w', 8081, 'N/A', 'local', 'test', endTime='N/A', status='INITIALIZED')
        self.assertTrue('xOpEwA5w' in experiment.get_all_experiments())
        experiment.remove_experiment('xOpEwA5w')
        self.assertFalse('xOpEwA5w' in experiment.get_all_experiments())

    def test_get_config(self):
        config = Config('xOpEwA5w', HOME_PATH)
        self.assertEqual(config.get_config()['experimentName'], 'test_config')

if __name__ == '__main__':
    main()
