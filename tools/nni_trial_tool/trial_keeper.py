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
import sys
import select
import json
from pyhdfs import HdfsClient
import pkg_resources
from .rest_utils import rest_post
from .url_utils import gen_send_stdout_url, gen_send_version_url

from .constants import HOME_DIR, LOG_DIR, NNI_PLATFORM, STDOUT_FULL_PATH, STDERR_FULL_PATH
from .hdfsClientUtility import copyDirectoryToHdfs, copyHdfsDirectoryToLocal
from .log_utils import LogType, nni_log, RemoteLogger, PipeLogReader, StdOutputType

logger = logging.getLogger('trial_keeper')
regular = re.compile('v?(?P<version>[0-9](\.[0-9]){0,1}).*')

def main_loop(args):
    '''main loop logic for trial keeper'''
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    stdout_file = open(STDOUT_FULL_PATH, 'a+')
    stderr_file = open(STDERR_FULL_PATH, 'a+')
    trial_keeper_syslogger = RemoteLogger(args.nnimanager_ip, args.nnimanager_port, 'trial_keeper', StdOutputType.Stdout, args.log_collection)
    # redirect trial keeper's stdout and stderr to syslog
    trial_syslogger_stdout = RemoteLogger(args.nnimanager_ip, args.nnimanager_port, 'trial', StdOutputType.Stdout, args.log_collection)
    sys.stdout = sys.stderr = trial_keeper_syslogger
    # backward compatibility
    hdfs_host = None
    hdfs_output_dir = None
    if args.hdfs_host:
        hdfs_host = args.hdfs_host
    elif args.pai_hdfs_host:
        hdfs_host = args.pai_hdfs_host
    if args.hdfs_output_dir:
        hdfs_output_dir = args.hdfs_output_dir
    elif args.pai_hdfs_output_dir:
        hdfs_output_dir = args.pai_hdfs_output_dir

    if hdfs_host is not None and args.nni_hdfs_exp_dir is not None:
        try:
            if args.webhdfs_path:
                hdfs_client = HdfsClient(hosts='{0}:80'.format(hdfs_host), user_name=args.pai_user_name, webhdfs_path=args.webhdfs_path, timeout=5)
            else:
                # backward compatibility
                hdfs_client = HdfsClient(hosts='{0}:{1}'.format(hdfs_host, '50070'), user_name=args.pai_user_name, timeout=5)
        except Exception as e:
            nni_log(LogType.Error, 'Create HDFS client error: ' + str(e))
            raise e
        copyHdfsDirectoryToLocal(args.nni_hdfs_exp_dir, os.getcwd(), hdfs_client)

    # Notice: We don't appoint env, which means subprocess wil inherit current environment and that is expected behavior
    log_pipe_stdout = trial_syslogger_stdout.get_pipelog_reader()
    process = Popen(args.trial_command, shell = True, stdout = log_pipe_stdout, stderr = log_pipe_stdout)
    nni_log(LogType.Info, 'Trial keeper spawns a subprocess (pid {0}) to run command: {1}'.format(process.pid, shlex.split(args.trial_command)))

    while True:
        retCode = process.poll()
        # child worker process exits and all stdout data is read
        if retCode is not None and log_pipe_stdout.set_process_exit() and log_pipe_stdout.is_read_completed == True:
            nni_log(LogType.Info, 'subprocess terminated. Exit code is {}. Quit'.format(retCode))
            if hdfs_output_dir is not None:
                # Copy local directory to hdfs for OpenPAI
                nni_local_output_dir = os.environ['NNI_OUTPUT_DIR']
                try:
                    if copyDirectoryToHdfs(nni_local_output_dir, hdfs_output_dir, hdfs_client):
                        nni_log(LogType.Info, 'copy directory from {0} to {1} success!'.format(nni_local_output_dir, hdfs_output_dir))
                    else:
                        nni_log(LogType.Info, 'copy directory from {0} to {1} failed!'.format(nni_local_output_dir, hdfs_output_dir))
                except Exception as e:
                    nni_log(LogType.Error, 'HDFS copy directory got exception: ' + str(e))
                    raise e

            ## Exit as the retCode of subprocess(trial)
            exit(retCode)
            break

        time.sleep(2)

def trial_keeper_help_info(*args):
    print('please run --help to see guidance')

def check_version(args):
    try:
        trial_keeper_version = pkg_resources.get_distribution('nni').version
    except pkg_resources.ResolutionError as err:
        #package nni does not exist, try nni-tool package
        nni_log(LogType.Error, 'Package nni does not exist!')
        os._exit(1)
    if not args.nni_manager_version:
        # skip version check
        nni_log(LogType.Warning, 'Skipping version check!')
    else:
        try:
            trial_keeper_version = regular.search(trial_keeper_version).group('version')
            nni_log(LogType.Info, 'trial_keeper_version is {0}'.format(trial_keeper_version))
            nni_manager_version = regular.search(args.nni_manager_version).group('version')
            nni_log(LogType.Info, 'nni_manager_version is {0}'.format(nni_manager_version))
            log_entry = {}
            if trial_keeper_version != nni_manager_version:
                nni_log(LogType.Error, 'Version does not match!')
                error_message = 'NNIManager version is {0}, TrialKeeper version is {1}, NNI version does not match!'.format(nni_manager_version, trial_keeper_version)
                log_entry['tag'] = 'VCFail'
                log_entry['msg'] = error_message
                rest_post(gen_send_version_url(args.nnimanager_ip, args.nnimanager_port), json.dumps(log_entry), 10, False)
                os._exit(1)
            else:
                nni_log(LogType.Info, 'Version match!')
                log_entry['tag'] = 'VCSuccess'
                rest_post(gen_send_version_url(args.nnimanager_ip, args.nnimanager_port), json.dumps(log_entry), 10, False)
        except AttributeError as err:
            nni_log(LogType.Error, err)

if __name__ == '__main__':
    '''NNI Trial Keeper main function'''
    PARSER = argparse.ArgumentParser()
    PARSER.set_defaults(func=trial_keeper_help_info)
    PARSER.add_argument('--trial_command', type=str, help='Command to launch trial process')
    PARSER.add_argument('--nnimanager_ip', type=str, default='localhost', help='NNI manager rest server IP')
    PARSER.add_argument('--nnimanager_port', type=str, default='8081', help='NNI manager rest server port')
    PARSER.add_argument('--pai_hdfs_output_dir', type=str, help='the output dir of pai_hdfs') # backward compatibility
    PARSER.add_argument('--hdfs_output_dir', type=str, help='the output dir of hdfs')
    PARSER.add_argument('--pai_hdfs_host', type=str, help='the host of pai_hdfs') # backward compatibility
    PARSER.add_argument('--hdfs_host', type=str, help='the host of hdfs')
    PARSER.add_argument('--pai_user_name', type=str, help='the username of hdfs')
    PARSER.add_argument('--nni_hdfs_exp_dir', type=str, help='nni experiment directory in hdfs')
    PARSER.add_argument('--webhdfs_path', type=str, help='the webhdfs path used in webhdfs URL')
    PARSER.add_argument('--nni_manager_version', type=str, help='the nni version transmitted from nniManager')
    PARSER.add_argument('--log_collection', type=str, help='set the way to collect log in trialkeeper')
    args, unknown = PARSER.parse_known_args()
    if args.trial_command is None:
        exit(1)
    check_version(args)
    try:
        main_loop(args)
    except SystemExit as se:
        nni_log(LogType.Info, 'NNI trial keeper exit with code {}'.format(se.code))
        os._exit(se.code)
    except Exception as e:
        nni_log(LogType.Error, 'Exit trial keeper with code 1 because Exception: {} is catched'.format(str(e)))
        os._exit(1)

