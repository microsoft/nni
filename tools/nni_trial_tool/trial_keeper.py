# ============================================================================================================================== #
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================================================================== #

import argparse
import sys
import os
from subprocess import Popen, PIPE
import time
import logging
import shlex
import re
from pyhdfs import HdfsClient

from .constants import HOME_DIR, LOG_DIR, NNI_PLATFORM, STDOUT_FULL_PATH, STDERR_FULL_PATH
from .hdfsClientUtility import copyDirectoryToHdfs
from .log_utils import LogType, nni_log
from .metrics_reader import read_experiment_metrics

logger = logging.getLogger('trial_keeper')

def main_loop(args):
    '''main loop logic for trial keeper'''
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    stdout_file = open(STDOUT_FULL_PATH, 'a+')
    stderr_file = open(STDERR_FULL_PATH, 'a+')
    # Notice: We don't appoint env, which means subprocess wil inherit current environment and that is expected behavior
    process = Popen(args.trial_command, shell = True, stdout = stdout_file, stderr = stderr_file)
    nni_log(LogType.Info, 'Trial keeper spawns a subprocess (pid {0}) to run command: {1}'.format(process.pid, shlex.split(args.trial_command)))
    
    while True:
        retCode = process.poll()
        ## Read experiment metrics, to avoid missing metrics
        read_experiment_metrics(args.nnimanager_ip, args.nnimanager_port)
        
        if retCode is not None:
            nni_log(LogType.Info, 'subprocess terminated. Exit code is {}. Quit'.format(retCode))
            if NNI_PLATFORM == 'pai':
                # Copy local directory to hdfs for OpenPAI
                nni_local_output_dir = os.environ['NNI_OUTPUT_DIR']
                try:
                    hdfs_client = HdfsClient(hosts='{0}:{1}'.format(args.pai_hdfs_host, '50070'), user_name=args.pai_user_name, timeout=5)
                    if copyDirectoryToHdfs(nni_local_output_dir, args.pai_hdfs_output_dir, hdfs_client):
                        nni_log(LogType.Info, 'copy directory from {0} to {1} success!'.format(nni_local_output_dir, args.pai_hdfs_output_dir))
                    else:
                        nni_log(LogType.Info, 'copy directory from {0} to {1} failed!'.format(nni_local_output_dir, args.pai_hdfs_output_dir))
                except Exception as e:
                    nni_log(LogType.Error, 'HDFS copy directory got exception: ' + str(e))
                    raise e

            ## Exit as the retCode of subprocess(trial)
            exit(retCode)
            break

        time.sleep(2)

def trial_keeper_help_info(*args):
    print('please run --help to see guidance')

if __name__ == '__main__':
    '''NNI Trial Keeper main function'''
    PARSER = argparse.ArgumentParser()
    PARSER.set_defaults(func=trial_keeper_help_info)
    PARSER.add_argument('--trial_command', type=str, help='Command to launch trial process')
    PARSER.add_argument('--nnimanager_ip', type=str, default='localhost', help='NNI manager rest server IP')
    PARSER.add_argument('--nnimanager_port', type=str, default='8081', help='NNI manager rest server port')
    PARSER.add_argument('--pai_hdfs_output_dir', type=str, help='the output dir of hdfs')
    PARSER.add_argument('--pai_hdfs_host', type=str, help='the host of hdfs')
    PARSER.add_argument('--pai_user_name', type=str, help='the username of hdfs')
    args, unknown = PARSER.parse_known_args()
    if args.trial_command is None:
        exit(1)

    try:
        main_loop(args)
    except SystemExit as se:
        nni_log(LogType.Info, 'NNI trial keeper exit with code {}'.format(se.code))
        sys.exit(se.code)
    except Exception as e:
        nni_log(LogType.Error, 'Exit trial keeper with code 1 because Exception: {} is catched'.format(str(e)))
        sys.exit(1)

