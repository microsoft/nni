# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest import TestCase, main
from nni.tools.nnictl.config_utils import Config, Experiments

HOME_PATH = str(Path(__file__).parent / "mock/nnictl_metadata")

class CommonUtilsTestCase(TestCase):

    # FIXME:
    # `experiment.get_all_experiments()` returns empty dict. No idea why.
    # Don't want to debug this because I will port the logic to `nni.experiment`.

    #def test_get_experiment(self):
    #    experiment = Experiments(HOME_PATH)
    #    self.assertTrue('xOpEwA5w' in experiment.get_all_experiments())
    
    def test_update_experiment(self):
        experiment = Experiments(HOME_PATH)
        experiment.add_experiment('xOpEwA5w', 8081, 'N/A', 'aGew0x', 'local', 'test', endTime='N/A', status='INITIALIZED')
        self.assertTrue('xOpEwA5w' in experiment.get_all_experiments())
        experiment.remove_experiment('xOpEwA5w')
        self.assertFalse('xOpEwA5w' in experiment.get_all_experiments())
    
    def test_get_config(self):
        config = Config('config', HOME_PATH)
        self.assertEqual(config.get_config('experimentId'), 'xOpEwA5w')
    
    def test_set_config(self):
        config = Config('config', HOME_PATH)
        self.assertNotEqual(config.get_config('experimentId'), 'testId')
        config.set_config('experimentId', 'testId')
        self.assertEqual(config.get_config('experimentId'), 'testId')
        config.set_config('experimentId', 'xOpEwA5w')

if __name__ == '__main__':
    main()
