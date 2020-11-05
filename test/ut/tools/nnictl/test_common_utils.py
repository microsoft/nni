# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
import sys
from unittest import TestCase, main, skipIf

from mock.restful_server import init_response

from nni.tools.nnictl.command_utils import kill_command
from nni.tools.nnictl.common_utils import get_yml_content, get_json_content, detect_process

cwd = Path(__file__).parent

class CommonUtilsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        init_response()

    def test_get_yml(self):
        yml_path = cwd / 'config_files/test_files/test_yaml.yml'
        content = get_yml_content(str(yml_path))
        self.assertEqual(content, {'field':'test'})

    def test_get_json(self):
        json_path = cwd / 'config_files/test_files/test_json.json'
        content = get_json_content(str(json_path))
        self.assertEqual(content, {'field':'test'})

    @skipIf(sys.platform == 'win32', 'FIXME: Fails randomly on Windows, cannot reproduce locally')
    def test_detect_process(self):
        if sys.platform == 'win32':
            cmds = ['timeout', '360000']
        else:
            cmds = ['sleep', '360000']
        process = Popen(cmds, stdout=PIPE, stderr=STDOUT)
        self.assertTrue(detect_process(process.pid))
        kill_command(process.pid)

if __name__ == '__main__':
    main()
