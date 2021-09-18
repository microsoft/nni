# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import ctypes
import json
import logging
import os
import re
import shlex
import sys
import threading
import time
from subprocess import Popen

import pkg_resources
from pyhdfs import HdfsClient

from .constants import (LOG_DIR, MULTI_PHASE, NNI_EXP_ID, NNI_PLATFORM,
                        NNI_SYS_DIR, NNI_TRIAL_JOB_ID)
from .hdfsClientUtility import (copyDirectoryToHdfs, copyHdfsDirectoryToLocal,
                                copyHdfsFileToLocal)
from .log_utils import LogType, RemoteLogger, StdOutputType, nni_log
from .rest_utils import rest_get, rest_post
from .url_utils import gen_parameter_meta_url, gen_send_version_url

logger = logging.getLogger('trial_keeper')
regular = re.compile('v?(?P<version>[0-9](\.[0-9]){0,1}).*')

_hdfs_client = None
_trial_process = None


def get_hdfs_client(args):
    global _hdfs_client

    if _hdfs_client is not None:
        return _hdfs_client
    # backward compatibility
    hdfs_host = None

    if args.hdfs_host:
        hdfs_host = args.hdfs_host
    elif args.pai_hdfs_host:
        hdfs_host = args.pai_hdfs_host
    else:
        return None

    if hdfs_host is not None and args.nni_hdfs_exp_dir is not None:
        try:
            if args.webhdfs_path:
                _hdfs_client = HdfsClient(hosts='{0}:80'.format(hdfs_host), user_name=args.pai_user_name,
                                          webhdfs_path=args.webhdfs_path, timeout=5)
            else:
                # backward compatibility
                _hdfs_client = HdfsClient(hosts='{0}:{1}'.format(hdfs_host, '50070'), user_name=args.pai_user_name,
                                          timeout=5)
        except Exception as e:
            nni_log(LogType.Error, 'Create HDFS client error: ' + str(e))
            raise e
    return _hdfs_client


def main_loop(args):
    '''main loop logic for trial keeper'''
    global _trial_process

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    trial_keeper_syslogger = RemoteLogger(args.nnimanager_ip, args.nnimanager_port, 'trial_keeper',
                                          StdOutputType.Stdout, args.log_collection)
    # redirect trial keeper's stdout and stderr to syslog
    trial_syslogger_stdout = RemoteLogger(args.nnimanager_ip, args.nnimanager_port, 'trial', StdOutputType.Stdout,
                                          args.log_collection)
    sys.stdout = sys.stderr = trial_keeper_syslogger
    hdfs_output_dir = None

    if args.hdfs_output_dir:
        hdfs_output_dir = args.hdfs_output_dir
    elif args.pai_hdfs_output_dir:
        hdfs_output_dir = args.pai_hdfs_output_dir

    hdfs_client = get_hdfs_client(args)

    if hdfs_client is not None:
        copyHdfsDirectoryToLocal(args.nni_hdfs_exp_dir, os.getcwd(), hdfs_client)

    if args.job_id_file:
        with open(args.job_id_file, 'w') as job_file:
            job_file.write("%d" % os.getpid())

    # Notice: We don't appoint env, which means subprocess wil inherit current environment and that is expected behavior
    log_pipe_stdout = trial_syslogger_stdout.get_pipelog_reader()
    if sys.platform == 'win32':
        _trial_process = Popen(args.trial_command, shell=True, stdout=log_pipe_stdout, stderr=log_pipe_stdout)
    else:
        _trial_process = Popen(args.trial_command, shell=True, stdout=log_pipe_stdout, stderr=log_pipe_stdout, preexec_fn=os.setsid)
    nni_log(LogType.Info, 'Trial keeper spawns a subprocess (pid {0}) to run command: {1}'.format(_trial_process.pid,
                                                                                                  shlex.split(
                                                                                                      args.trial_command)))

    while True:
        retCode = _trial_process.poll()
        # child worker process exits and all stdout data is read
        if retCode is not None and log_pipe_stdout.set_process_exit() and log_pipe_stdout.is_read_completed == True:
            # In Windows, the retCode -1 is 4294967295. It's larger than c_long, and raise OverflowError.
            # So covert it to int32.
            retCode = ctypes.c_long(retCode).value
            nni_log(LogType.Info, 'subprocess terminated. Exit code is {}. Quit'.format(retCode))
            if hdfs_output_dir is not None:
                # Copy local directory to hdfs for OpenPAI
                nni_local_output_dir = os.environ['NNI_OUTPUT_DIR']
                try:
                    if copyDirectoryToHdfs(nni_local_output_dir, hdfs_output_dir, hdfs_client):
                        nni_log(LogType.Info,
                                'copy directory from {0} to {1} success!'.format(nni_local_output_dir, hdfs_output_dir))
                    else:
                        nni_log(LogType.Info,
                                'copy directory from {0} to {1} failed!'.format(nni_local_output_dir, hdfs_output_dir))
                except Exception as e:
                    nni_log(LogType.Error, 'HDFS copy directory got exception: ' + str(e))
                    raise e

            # Exit as the retCode of subprocess(trial)
            exit(retCode)
            break

        time.sleep(2)


def trial_keeper_help_info(*args):
    print('please run --help to see guidance')


def check_version(args):
    try:
        trial_keeper_version = pkg_resources.get_distribution('nni').version
    except pkg_resources.ResolutionError:
        # package nni does not exist, try nni-tool package
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
                nni_log(LogType.Warning, 'Version does not match!')
                error_message = 'NNIManager version is {0}, TrialKeeper version is {1}, NNI version does not match!'.format(
                    nni_manager_version, trial_keeper_version)
                log_entry['tag'] = 'VCFail'
                log_entry['msg'] = error_message
                rest_post(gen_send_version_url(args.nnimanager_ip, args.nnimanager_port), json.dumps(log_entry), 10,
                          False)
            else:
                nni_log(LogType.Info, 'Version match!')
                log_entry['tag'] = 'VCSuccess'
                rest_post(gen_send_version_url(args.nnimanager_ip, args.nnimanager_port), json.dumps(log_entry), 10,
                          False)
        except AttributeError as err:
            nni_log(LogType.Error, err)


def is_multi_phase():
    return MULTI_PHASE and (MULTI_PHASE in ['True', 'true'])


def download_parameter(meta_list, args):
    """
    Download parameter file to local working directory.
    meta_list format is defined in paiJobRestServer.ts
    example meta_list:
    [
        {"experimentId":"yWFJarYa","trialId":"UpPkl","filePath":"/chec/nni/experiments/yWFJarYa/trials/UpPkl/parameter_1.cfg"},
        {"experimentId":"yWFJarYa","trialId":"aIUMA","filePath":"/chec/nni/experiments/yWFJarYa/trials/aIUMA/parameter_1.cfg"}
    ]
    """
    nni_log(LogType.Debug, str(meta_list))
    nni_log(LogType.Debug,
            'NNI_SYS_DIR: {}, trial Id: {}, experiment ID: {}'.format(NNI_SYS_DIR, NNI_TRIAL_JOB_ID, NNI_EXP_ID))
    nni_log(LogType.Debug, 'NNI_SYS_DIR files: {}'.format(os.listdir(NNI_SYS_DIR)))
    for meta in meta_list:
        if meta['experimentId'] == NNI_EXP_ID and meta['trialId'] == NNI_TRIAL_JOB_ID:
            param_fp = os.path.join(NNI_SYS_DIR, os.path.basename(meta['filePath']))
            if not os.path.exists(param_fp):
                hdfs_client = get_hdfs_client(args)
                copyHdfsFileToLocal(meta['filePath'], param_fp, hdfs_client, override=False)


def fetch_parameter_file(args):
    class FetchThread(threading.Thread):
        def __init__(self, args):
            super(FetchThread, self).__init__()
            self.args = args

        def run(self):
            uri = gen_parameter_meta_url(self.args.nnimanager_ip, self.args.nnimanager_port)
            nni_log(LogType.Info, uri)

            while True:
                res = rest_get(uri, 10)
                nni_log(LogType.Debug, 'status code: {}'.format(res.status_code))
                if res.status_code == 200:
                    meta_list = res.json()
                    download_parameter(meta_list, self.args)
                else:
                    nni_log(LogType.Warning, 'rest response: {}'.format(str(res)))
                time.sleep(5)

    fetch_file_thread = FetchThread(args)
    fetch_file_thread.start()


def _set_adaptdl_signal_handler():
    import signal
    global _trial_process
    def _handler(signum, frame):
        nni_log(LogType.Info, "RECEIVED SIGNAL {}".format(signum))
        nni_log(LogType.Debug, "TRIAL PROCESS ID {}".format(_trial_process.pid))
        if _trial_process and (signum == signal.SIGTERM or signum == signal.SIGINT):
            os.killpg(os.getpgid(_trial_process.pid), signal.SIGINT)
            os.waitpid(_trial_process.pid, 0)
        exit(1)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


if __name__ == '__main__':
    '''NNI Trial Keeper main function'''
    PARSER = argparse.ArgumentParser()
    PARSER.set_defaults(func=trial_keeper_help_info)
    PARSER.add_argument('--trial_command', type=str, help='Command to launch trial process')
    PARSER.add_argument('--nnimanager_ip', type=str, default='localhost', help='NNI manager rest server IP')
    PARSER.add_argument('--nnimanager_port', type=str, default='8081', help='NNI manager rest server port')
    PARSER.add_argument('--pai_hdfs_output_dir', type=str, help='the output dir of pai_hdfs')  # backward compatibility
    PARSER.add_argument('--hdfs_output_dir', type=str, help='the output dir of hdfs')
    PARSER.add_argument('--pai_hdfs_host', type=str, help='the host of pai_hdfs')  # backward compatibility
    PARSER.add_argument('--hdfs_host', type=str, help='the host of hdfs')
    PARSER.add_argument('--pai_user_name', type=str, help='the username of hdfs')
    PARSER.add_argument('--nni_hdfs_exp_dir', type=str, help='nni experiment directory in hdfs')
    PARSER.add_argument('--webhdfs_path', type=str, help='the webhdfs path used in webhdfs URL')
    PARSER.add_argument('--nni_manager_version', type=str, help='the nni version transmitted from nniManager')
    PARSER.add_argument('--log_collection', type=str, help='set the way to collect log in trialkeeper')
    PARSER.add_argument('--job_id_file', type=str, help='set job id file for operating and monitoring job.')
    args, unknown = PARSER.parse_known_args()
    if args.trial_command is None:
        exit(1)
    check_version(args)
    try:
        if NNI_PLATFORM == 'adl':
            _set_adaptdl_signal_handler()
        main_loop(args)
    except SystemExit as se:
        nni_log(LogType.Info, 'NNI trial keeper exit with code {}'.format(se.code))
        os._exit(se.code)
    except Exception as e:
        nni_log(LogType.Error, 'Exit trial keeper with code 1 because Exception: {} is catched'.format(str(e)))
        os._exit(1)
