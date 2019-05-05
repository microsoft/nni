# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nni
import os
import sys
from subprocess import call
from .constants import PACKAGE_REQUIREMENTS
from .common_utils import print_normal, print_error
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
    
