# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests
import responses
import glob
import psutil
from unittest import TestCase, main
from nni_cmd.common_utils import get_yml_content, get_json_content, \
 detect_process, detect_port, get_user, check_tensorboard_version, generate_temp_dir, \
 print_error, print_green
from mock.restful_server import init_response
from subprocess import Popen, call, check_call, CalledProcessError, PIPE, STDOUT
from nni_cmd.config_utils import Config, Experiments

HOME_PATH = "./tests/mock/nnictl_metadata"
class CommonUtilsTestCase(TestCase):

    def test_get_experiment(self):
        experiment = Experiments(HOME_PATH)
        self.assertTrue('xOpEwA5w' in experiment.get_all_experiments())
    
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
