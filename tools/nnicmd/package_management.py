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
from subprocess import call
from .constants import PACKAGE_REQUIREMENTS
from .common_utils import print_normal

def manage_smac(mode):
    if mode == 'install':
        for package in PACKAGE_REQUIREMENTS.get('SMAC').get('install'):
            print_normal('installing ' + package)
            cmds = ['pip3', 'install', '--user', package]
            call(cmds)
    else:
        for package in PACKAGE_REQUIREMENTS.get('SMAC').get('uninstall'):
            print_normal('uninstalling ' + package)
            cmds = ['pip3', 'uninstall', '-y', package]
            call(cmds)

def package_install(args):
    '''install packages'''
    if args.name == 'SMAC':
        manage_smac('install')
    
def package_uninstall(args):
    '''uninstall packages'''
    if args.name == 'SMAC':
        manage_smac('uninstall')

def package_show(args):
    '''show all packages'''
    print(' '.join(PACKAGE_REQUIREMENTS.keys()))