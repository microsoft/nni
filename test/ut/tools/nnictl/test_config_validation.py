# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
from unittest import TestCase, main
from schema import SchemaError
from nni.tools.nnictl.launcher_utils import validate_all_content
from nni.tools.nnictl.nnictl_utils import get_yml_content
from nni.tools.nnictl.common_utils import print_error, print_green

class ConfigValidationTestCase(TestCase):
    def test_valid_config(self):
        file_names = glob.glob('./config_files/valid/*.yml')
        for fn in file_names:
            experiment_config = get_yml_content(fn)
            validate_all_content(experiment_config, fn)
            print_green('config file:', fn, 'validation success!')

    def test_invalid_config(self):
        file_names = glob.glob('./config_files/invalid/*.yml')
        for fn in file_names:
            experiment_config = get_yml_content(fn)
            try:
                validate_all_content(experiment_config, fn)
                print_error('config file:', fn,'Schema error should be raised for invalid config file!')
                assert False
            except SchemaError as e:
                print_green('config file:', fn, 'Expected error catched:', e)

if __name__ == '__main__':
    main()
