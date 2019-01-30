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

import os

REST_PORT = 51188

HOME_DIR = os.path.join(os.environ['HOME'], 'nni')

METADATA_DIR = os.path.join(HOME_DIR, 'nnictl')

METADATA_FULL_PATH = os.path.join(METADATA_DIR, 'metadata')

LOG_DIR = os.path.join(HOME_DIR, 'nnictl', 'log')

STDOUT_FULL_PATH = os.path.join(LOG_DIR, 'stdout')

STDERR_FULL_PATH = os.path.join(LOG_DIR, 'stderr')

ERROR_INFO = 'Error: %s'

NORMAL_INFO = 'Info: %s'

WARNING_INFO = 'Waining: %s'

EXPERIMENT_SUCCESS_INFO = 'Start experiment success! The experiment id is %s, and the restful server post is %s.\n' \
                          'You can use these commands to get more information about this experiment:\n' \
                          '         commands                       description\n' \
                          '1. nnictl experiment show        show the information of experiments\n' \
                          '2. nnictl trial ls               list all of trial jobs\n' \
                          '3. nnictl stop                   stop a experiment\n' \
                          '4. nnictl trial kill             kill a trial job by id\n' \
                          '5. nnictl --help                 get help information about nnictl\n' \
                          '6. nnictl webui url              get the url of web ui'
