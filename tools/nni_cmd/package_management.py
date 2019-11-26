# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import nni
from .constants import PACKAGE_REQUIREMENTS
from .common_utils import print_error
from .command_utils import install_requirements_command

def process_install(package_name):
    if PACKAGE_REQUIREMENTS.get(package_name) is None:
        print_error('{0} is not supported!' % package_name)
    else:
        requirements_path = os.path.join(nni.__path__[0], PACKAGE_REQUIREMENTS[package_name])
        install_requirements_command(requirements_path)

def package_install(args):
    '''install packages'''
    process_install(args.name)

def package_show(args):
    '''show all packages'''
    print(' '.join(PACKAGE_REQUIREMENTS.keys()))

