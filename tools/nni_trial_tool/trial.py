# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
import os
import shlex
import tarfile
import random
from datetime import datetime
from subprocess import Popen

import psutil

from .log_utils import LogType, RemoteLogger, StdOutputType, nni_log

trial_output_path_name = ".nni"


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
            if (self.args.node_count > 1):
                while True:
                    exit_code_file_name = "%s_%s" % (exit_code_file_name, random.randint(0, 10000))
                    if not os.path.exists(exit_code_file_name):
                        break
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
