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
import traceback
import tarfile
import psutil
from datetime import datetime
from subprocess import Popen

import pkg_resources

idle_timeout_seconds = 10 * 60

logger = logging.getLogger('trial_runner')
regular = re.compile('v?(?P<version>[0-9](\.[0-9]){0,1}).*')
trial_output_path_name = ".nni"
trial_runner_syslogger = None


class Trial:
    def __init__(self, args, data):
        self.process = None
        self.data = data
        self.args = args
        self.trial_syslogger_stdout = None

        global NNI_TRIAL_JOB_ID
        self.id = data["trialId"]
        if self.id is None:
            raise Exception("trial_id is not found in %s" % data)
        os.environ['NNI_TRIAL_JOB_ID'] = self.id
        NNI_TRIAL_JOB_ID = self.id

    def run(self):
        # redirect trial runner's stdout and stderr to syslog
        self.trial_syslogger_stdout = RemoteLogger(self.args.nnimanager_ip, self.args.nnimanager_port, 'trial', StdOutputType.Stdout,
                                                   self.args.log_collection, self.id)

        nni_log(LogType.Info, "start to run trial %s" % self.id)

        trial_working_dir = os.path.realpath(os.path.join(os.curdir, "..", "..", "trials", self.id))

        os.environ['NNI_TRIAL_SEQ_ID'] = str(self.data["sequenceId"])
        os.environ['NNI_OUTPUT_DIR'] = os.path.join(trial_working_dir, "nnioutput")
        os.environ['NNI_SYS_DIR'] = trial_working_dir

        self.trial_output_dir = os.path.join(trial_working_dir, trial_output_path_name)
        os.makedirs(self.trial_output_dir, exist_ok=True)
        trial_code_dir = os.path.join(trial_working_dir, "code")
        os.makedirs(trial_code_dir, exist_ok=True)

        # prepare code
        with tarfile.open(os.path.join("..", "nni-code.tar.gz"), "r:gz") as tar:
            tar.extractall(trial_code_dir)

        # save parameters
        nni_log(LogType.Info, 'saving parameter %s' % self.data["parameter"]["value"])
        parameter_file_name = os.path.join(trial_working_dir, "parameter.cfg")
        with open(parameter_file_name, "w") as parameter_file:
            parameter_file.write(self.data["parameter"]["value"])

        # Notice: We don't appoint env, which means subprocess wil inherit current environment and that is expected behavior
        self.log_pipe_stdout = self.trial_syslogger_stdout.get_pipelog_reader()
        self.process = Popen(self.args.trial_command, shell=True, stdout=self.log_pipe_stdout,
                             stderr=self.log_pipe_stdout, cwd=trial_code_dir, env=os.environ)
        nni_log(LogType.Info, 'Trial runner spawns a subprocess (pid {0}) to run command: {1}'.
                format(self.process.pid, shlex.split(self.args.trial_command)))

    def is_running(self):
        if (self.process is None):
            return False

        retCode = self.process.poll()
        # child worker process exits and all stdout data is read
        if retCode is not None and self.log_pipe_stdout.set_process_exit() and self.log_pipe_stdout.is_read_completed == True:
            # In Windows, the retCode -1 is 4294967295. It's larger than c_long, and raise OverflowError.
            # So covert it to int32.
            retCode = ctypes.c_long(retCode).value
            nni_log(LogType.Info, 'subprocess terminated. Exit code is {}. Quit'.format(retCode))

            # Exit as the retCode of subprocess(trial)
            exit_code_file_name = os.path.join(self.trial_output_dir, "code")
            with open(exit_code_file_name, "w") as exit_file:
                exit_file.write("%s %s" % (retCode, int(datetime.now().timestamp() * 1000)))
            self.cleanup()
            return False
        else:
            return True

    def kill(self, trial_id=None):
        if trial_id == self.id or trial_id is None:
            if self.process is not None:
                nni_log(LogType.Info, "killing trial %s" % self.id)
                for child in psutil.Process(self.process.pid).children(True):
                    child.kill()
                self.process.kill()
            self.cleanup()

    def cleanup(self):
        nni_log(LogType.Info, "clean up trial %s" % self.id)
        self.process = None
        if self.log_pipe_stdout is not None:
            self.log_pipe_stdout.set_process_exit()
            self.log_pipe_stdout = None
        if self.trial_syslogger_stdout is not None:
            self.trial_syslogger_stdout.close()
            self.trial_syslogger_stdout = None


def main_loop(args):
    '''main loop logic for trial runner'''
    idle_last_time = datetime.now()
    trial_runner_syslogger = RemoteLogger(args.nnimanager_ip, args.nnimanager_port, 'trial_runner',
                                          StdOutputType.Stdout, args.log_collection, args.runner_id)
    sys.stdout = sys.stderr = trial_runner_syslogger
    trial = None

    try:
        # command loop
        while True:
            command_type, command_data = receive()
            if command_type == CommandType.NewTrialJob:
                if trial is not None:
                    raise Exception('trial %s is running already, cannot start a new one' % trial.trial_id)
                trial = Trial(args, command_data)
                trial.run()
            elif command_type == CommandType.KillTrialJob:
                if trial is not None:
                    trial.kill(command_data)
            elif command_type is not None:
                raise Exception("unknown command %s" % command_type)

            if trial is not None and trial.is_running():
                idle_last_time = datetime.now()
            else:
                trial = None

            if (datetime.now() - idle_last_time).seconds > idle_timeout_seconds:
                nni_log(LogType.Info, "trial runner is idle more than {0} seconds, so exit.".format(
                    idle_timeout_seconds))
                break
            time.sleep(1)
    except Exception as ex:
        nni_log(LogType.Error, ex)
    finally:
        if trial is not None:
            trial.kill()

    trial_runner_syslogger.close()
    trial_runner_syslogger = None


def trial_runner_help_info(*args):
    print('please run --help to see guidance')


def check_version(args):
    try:
        trial_runner_version = pkg_resources.get_distribution('nni').version
    except pkg_resources.ResolutionError as err:
        # package nni does not exist, try nni-tool package
        nni_log(LogType.Error, 'Package nni does not exist!')
        os._exit(1)
    if not args.nni_manager_version:
        # skip version check
        nni_log(LogType.Warning, 'Skipping version check!')
    else:
        try:
            trial_runner_version = regular.search(trial_runner_version).group('version')
            nni_log(LogType.Info, 'trial_runner_version is {0}'.format(trial_runner_version))
            nni_manager_version = regular.search(args.nni_manager_version).group('version')
            nni_log(LogType.Info, 'nni_manager_version is {0}'.format(nni_manager_version))
            log_entry = {}
            if trial_runner_version != nni_manager_version:
                nni_log(LogType.Error, 'Version does not match!')
                error_message = 'NNIManager version is {0}, Trial runner version is {1}, NNI version does not match!'.format(
                    nni_manager_version, trial_runner_version)
                log_entry['tag'] = 'VCFail'
                log_entry['msg'] = error_message
                rest_post(gen_send_version_url(args.nnimanager_ip, args.nnimanager_port, args.runner_id), json.dumps(log_entry), 10,
                          False)
                os._exit(1)
            else:
                nni_log(LogType.Info, 'Version match!')
                log_entry['tag'] = 'VCSuccess'
                rest_post(gen_send_version_url(args.nnimanager_ip, args.nnimanager_port, args.runner_id), json.dumps(log_entry), 10,
                          False)
        except AttributeError as err:
            nni_log(LogType.Error, err)


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
                if res.status_code != 200:
                    nni_log(LogType.Warning, 'rest response: {}'.format(str(res)))
                time.sleep(2)

    fetch_file_thread = FetchThread(args)
    fetch_file_thread.start()


if __name__ == '__main__':

    '''NNI Trial Runner main function'''
    PARSER = argparse.ArgumentParser()
    PARSER.set_defaults(func=trial_runner_help_info)
    PARSER.add_argument('--trial_command', type=str, help='Command to launch trial process')
    PARSER.add_argument('--nnimanager_ip', type=str, help='NNI manager rest server IP')
    PARSER.add_argument('--nnimanager_port', type=str, help='NNI manager rest server port')
    PARSER.add_argument('--nni_manager_version', type=str, help='the nni version transmitted from nniManager')
    PARSER.add_argument('--log_collection', type=str, help='set the way to collect log in trial runner')
    args, unknown = PARSER.parse_known_args()

    setting_file = "../settings.json"
    if os.path.exists(setting_file):
        with open(setting_file, 'r') as fp:
            settings = json.load(fp)
    print("setting is {}".format(settings))

    args.exp_id = settings["experimentId"]
    args.platform = settings["platform"]
    args.runner_id = "runner_"+os.path.basename(os.path.realpath(os.path.curdir))

    if args.trial_command is None:
        args.trial_command = settings["command"]
    if args.nnimanager_ip is None:
        args.nnimanager_ip = settings["nniManagerIP"]
    if args.nnimanager_port is None:
        args.nnimanager_port = settings["nniManagerPort"]
    if args.nni_manager_version is None:
        args.nni_manager_version = settings["nniManagerVersion"]
    if args.log_collection is None:
        args.log_collection = settings["logCollection"]

    os.environ['NNI_OUTPUT_DIR'] = os.curdir + "/nnioutput"
    os.environ['NNI_PLATFORM'] = args.platform
    os.environ['NNI_SYS_DIR'] = os.curdir
    os.environ['NNI_EXP_ID'] = args.exp_id
    os.environ['MULTI_PHASE'] = "true"
    os.environ['NNI_TRIAL_JOB_ID'] = "runner"

    from .log_utils import LogType, RemoteLogger, StdOutputType, nni_log
    from .rest_utils import rest_get, rest_post
    from .url_utils import gen_parameter_meta_url, gen_send_version_url
    from .protocol import CommandType, receive

    nni_log(LogType.Info, "merged args is {}".format(args))

    if args.trial_command is None:
        nni_log(LogType.Error, "no command is found.")
        os._exit(1)
    check_version(args)
    try:
        main_loop(args)
    except SystemExit as se:
        nni_log(LogType.Info, 'NNI trial runner exit with code {}'.format(se.code))
        os._exit(se.code)
    finally:
        if trial_runner_syslogger is not None:
            if trial_runner_syslogger.pipeReader is not None:
                trial_runner_syslogger.pipeReader.set_process_exit()
            trial_runner_syslogger.close()

    # the process doesn't exit even main loop exit. So exit it explictly.
    os._exit(0)
