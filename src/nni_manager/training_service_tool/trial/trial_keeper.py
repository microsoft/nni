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

from .hdfsClientUtility import copyDirectoryToHdfs
from .constants import HOME_DIR, LOG_DIR, STDOUT_FULL_PATH, STDERR_FULL_PATH
from .metrics_reader import read_experiment_metrics

logger = logging.getLogger('trial_keeper')

def main_loop(args):
    '''main loop logic for trial keeper'''
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    stdout_file = open(STDOUT_FULL_PATH, 'a+')
    stderr_file = open(STDERR_FULL_PATH, 'a+')
    print(shlex.split(args.trial_command))
    # Notice: We don't appoint env, which means subprocess wil inherit current environment and that is expected behavior
    process = Popen(args.trial_command, shell = True, stdout = stdout_file, stderr = stderr_file)
    print('Subprocess pid is {}'.format(process.pid))
    print('Current cwd is {}'.format(os.getcwd()))
    while True:
        retCode = process.poll()
        ## Read experiment metrics, to avoid missing metrics
        read_experiment_metrics(args.nnimanager_ip)
        
        if retCode is not None:
            print('subprocess terminated. Exit code is {}. Quit'.format(retCode))
            #copy local directory to hdfs
            local_directory = os.environ['NNI_OUTPUT_DIR']
            trial_job_id = os.environ['NNI_TRIAL_JOB_ID']
            exp_id = os.environ['NNI_EXP_ID']
            hdfs_client = HdfsClient(hosts='{0}:{1}'.format(args.pai_hdfs_host, '50070'), user_name=args.pai_user_name)
            print(local_directory, args.pai_hdfs_output_dir)
            if copyDirectoryToHdfs(local_directory, args.pai_hdfs_output_dir, hdfs_client):
                print('copy directory success!')
            else:
                print('copy directory failed!')
            break
        else:
            print('subprocess pid: {} is still alive'.format(process.pid))

        time.sleep(2)

def trial_keeper_help_info(*args):
    print('please run --help to see guidance')

if __name__ == '__main__':
    '''NNI Trial Keeper main function'''
    PARSER = argparse.ArgumentParser()
    PARSER.set_defaults(func=trial_keeper_help_info)
    PARSER.add_argument('--trial_command', type=str, help='Command to launch trial process')
    PARSER.add_argument('--nnimanager_ip', type=str, default='localhost', help='NNI manager IP')
    PARSER.add_argument('--pai_hdfs_output_dir', type=str, help='the output dir of hdfs')
    PARSER.add_argument('--pai_hdfs_host', type=str, help='the host of hdfs')
    PARSER.add_argument('--pai_user_name', type=str, help='the username of hdfs')
    args, unknown = PARSER.parse_known_args()
    if args.trial_command is None:
        exit(1)

    try:
        main_loop(args)
    except:
        print('Exiting by user request')
        sys.exit(1)

