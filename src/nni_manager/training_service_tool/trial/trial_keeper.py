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
        if retCode is not None:
            print('subprocess terminated. Exit code is {}. Quit'.format(retCode))
            if 'NNI_OUTPUT_DIR' in os.environ and 'NNI_HDFS_OUTPUT_DIR' in os.environ and 'NNI_USER_NAME' in os.environ:
                local_directory = os.environ['NNI_OUTPUT_DIR']
                hdfs_output_dir = os.environ['NNI_HDFS_OUTPUT_DIR']
                nni_user_name = os.environ['NNI_USER_NAME']
                #get hdfs_host and hdfs_directory
                hdfs_host_pattern = 'hdfs://[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}:[0-9]{2,5}'
                hdfs_host = re.findall(hdfs_host_pattern, hdfs_output_dir)
                hdfs_directory = hdfs_output_dir.replace(hdfs_host[0], '')
                #get url_host
                url_host_pattern = '[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}'
                url_host = re.findall(url_host_pattern, hdfs_host[0])
                #init hdfs client
                if not os.path.isdir(local_directory):
                    raise Exception('Local Directory Error!')
                #get local folder name
                local_folder_name = local_directory.replace(os.path.dirname(local_directory), '')[1:]
                hdfs_output_dir_full = os.path.join(hdfs_directory, local_folder_name)
                hdfs_client = HdfsClient(hosts='{0}:{1}'.format(url_host[0], '50070'), user_name=nni_user_name)
                print(local_directory, hdfs_output_dir_full)
                if copyDirectoryToHdfs(local_directory, hdfs_output_dir_full, hdfs_client):
                    print('copy directory success!')
                else:
                    print('copy directory failed!')
            break
        else:
            print('subprocess pid: {} is still alive'.format(process.pid))
            read_experiment_metrics(args.nnimanager_ip)
        time.sleep(2)

def trial_keeper_help_info(*args):
    print('please run --help to see guidance')

if __name__ == '__main__':
    '''NNI Trial Keeper main function'''
    PARSER = argparse.ArgumentParser()
    PARSER.set_defaults(func=trial_keeper_help_info)
    PARSER.add_argument('--trial_command', type=str, help='Command to launch trial process')
    PARSER.add_argument('--nnimanager_ip', type=str, default='localhost', help='NNI manager IP')
    args, unknown = PARSER.parse_known_args()
    if args.trial_command is None:
        exit(1)

    try:
        main_loop(args)
    except:
        print('Exiting by user request')
        sys.exit(1)

