# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest import TestCase, main
from nni_cmd.common_utils import get_yml_content, get_json_content, detect_process
from mock.restful_server import init_response
from subprocess import Popen, PIPE, STDOUT
from nni_cmd.command_utils import kill_command

class CommonUtilsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        init_response()

    def test_get_yml(self):
        content = get_yml_content('./tests/config_files/test_files/test_yaml.yml')
        self.assertEqual(content, {'field':'test'})

    def test_get_json(self):
        content = get_json_content('./tests/config_files/test_files/test_json.json')
        self.assertEqual(content, {'field':'test'})

    def test_detect_process(self):
        cmds = ['sleep', '360000']
        process = Popen(cmds, stdout=PIPE, stderr=STDOUT)
        self.assertTrue(detect_process(process.pid))
        kill_command(process.pid)

if __name__ == '__main__':
    main()
