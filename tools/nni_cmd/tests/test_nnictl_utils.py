# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from mock.restful_server import init_response
from mock.experiment import create_mock_experiment, stop_mock_experiment, generate_args_parser, \
generate_args
from nni_cmd.nnictl_utils import get_experiment_time, get_experiment_status, \
check_experiment_id, parse_ids, get_config_filename, get_experiment_port, check_rest, \
trial_ls, list_experiment
from nni_cmd.rest_utils import check_rest_server_quick
from unittest import TestCase, main
import responses
import glob
import psutil
import requests

class CommonUtilsTestCase(TestCase):
    @classmethod
    def setUp(self):
        init_response()
        create_mock_experiment()

    @classmethod
    def tearDown(self):
        stop_mock_experiment()
    
    @responses.activate
    def test_get_experiment_time(self):
        self.assertEqual(get_experiment_time(8080), ('2020/09/17 15:14:55', '2020/09/17 15:15:10'))
        
    @responses.activate
    def test_get_experiment_status(self):
        self.assertEqual('RUNNING', get_experiment_status(8080))

    @responses.activate
    def test_check_experiment_id(self):
        parser = generate_args_parser()
        args = parser.parse_args(['xOpEwA5w'])
        self.assertEqual('xOpEwA5w', check_experiment_id(args))

    @responses.activate
    def test_parse_ids(self):
        parser = generate_args_parser()
        args = parser.parse_args(['xOpEwA5w'])
        self.assertEqual(['xOpEwA5w'], parse_ids(args))

    @responses.activate
    def test_get_config_file_name(self):
        args = generate_args()
        self.assertEqual('aGew0x', get_config_filename(args))
    
    @responses.activate
    def test_get_experiment_port(self):
        args = generate_args()
        self.assertEqual('8080', get_experiment_port(args))
    
    @responses.activate
    def test_check_rest(self):
        args = generate_args()
        #self.assertEqual(True, check_rest(args))

    @responses.activate
    def test_trial_ls(self):
        args = generate_args()
        self.assertEqual(trials[0]['id'], 'GPInz')


if __name__ == '__main__':
    main()
