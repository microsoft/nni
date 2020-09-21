
from mock.restful_server import init_response
from mock.experiment import create_mock_experiment, stop_mock_experiment, generate_args
from nni_cmd.nnictl_utils import get_experiment_time, get_experiment_status, \
check_experiment_id
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
        args = generate_args()
        print(check_experiment_id(args))
        

if __name__ == '__main__':
    main()
