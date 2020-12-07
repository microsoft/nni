# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
import os
import shlex
import tarfile
import time
from datetime import datetime
from subprocess import Popen

import psutil

from .log_utils import LogType, RemoteLogger, StdOutputType, nni_log
from .commands import CommandType

trial_output_path_name = ".nni"


class Trial:
    def __init__(self, args, data):
        self.process = None
        self.data = data
        self.args = args
        self.command_channel = args.command_channel
        self.trial_syslogger_stdout = None

        global NNI_TRIAL_JOB_ID
        self.id = data["trialId"]
        if self.id is None:
            raise Exception("trial_id is not found in %s" % data)
        os.environ['NNI_TRIAL_JOB_ID'] = self.id
        NNI_TRIAL_JOB_ID = self.id

        # for multiple nodes. If it's None, it means single node.
        self.node_id = args.node_id
        if self.node_id is None:
            self.name = self.id
        else:
            self.name = "%s_%s" % (self.id, self.node_id)

    def run(self):
        # redirect trial's stdout and stderr to syslog
        self.trial_syslogger_stdout = RemoteLogger(self.args.nnimanager_ip, self.args.nnimanager_port, 'trial', StdOutputType.Stdout,
                                                   self.args.log_collection, self.id, self.args.command_channel)

        nni_log(LogType.Info, "%s: start to run trial" % self.name)

        trial_working_dir = os.path.realpath(os.path.join(os.curdir, "..", "..", "trials", self.id))
        self.trial_output_dir = os.path.join(trial_working_dir, trial_output_path_name)
        trial_code_dir = os.path.join(trial_working_dir, "code")
        trial_nnioutput_dir = os.path.join(trial_working_dir, "nnioutput")

        environ = os.environ.copy()
        environ['NNI_TRIAL_SEQ_ID'] = str(self.data["sequenceId"])
        environ['NNI_OUTPUT_DIR'] = os.path.join(trial_working_dir, "nnioutput")
        environ['NNI_SYS_DIR'] = trial_working_dir
        self.working_dir = trial_working_dir

        # prepare code and parameters
        prepared_flag_file_name = os.path.join(trial_working_dir, "trial_prepared")
        if not os.path.exists(trial_working_dir):
            os.makedirs(trial_working_dir, exist_ok=True)

            os.makedirs(self.trial_output_dir, exist_ok=True)
            os.makedirs(trial_nnioutput_dir, exist_ok=True)
            # prepare code
            os.makedirs(trial_code_dir, exist_ok=True)
            with tarfile.open(os.path.join("..", "nni-code.tar.gz"), "r:gz") as tar:
                tar.extractall(trial_code_dir)

            # save parameters
            nni_log(LogType.Info, '%s: saving parameter %s' % (self.name, self.data["parameter"]["value"]))
            parameter_file_name = os.path.join(trial_working_dir, "parameter.cfg")
            with open(parameter_file_name, "w") as parameter_file:
                parameter_file.write(self.data["parameter"]["value"])

            # ready flag
            with open(prepared_flag_file_name, "w") as prepared_flag_file:
                prepared_flag_file.write("%s" % (int(datetime.now().timestamp() * 1000)))

        # make sure code prepared by other node.
        if self.node_id is not None:
            while True:
                if os.path.exists(prepared_flag_file_name):
                    break
                time.sleep(0.1)

        trial_command = self.args.trial_command

        gpuIndices = self.data.get('gpuIndices')
        if (gpuIndices is not None):
            trial_command = 'CUDA_VISIBLE_DEVICES="%s " %s' % (gpuIndices, trial_command)

        self.log_pipe_stdout = self.trial_syslogger_stdout.get_pipelog_reader()
        self.process = Popen(trial_command, shell=True, stdout=self.log_pipe_stdout,
                             stderr=self.log_pipe_stdout, cwd=trial_code_dir, env=dict(environ))
        nni_log(LogType.Info, '{0}: spawns a subprocess (pid {1}) to run command: {2}'.
                format(self.name, self.process.pid, shlex.split(trial_command)))

    def save_parameter_file(self, command_data):
        parameters = command_data["parameters"]
        file_index = int(parameters["index"])
        if file_index == 0:
            parameter_file_name = "parameter.cfg"
        else:
            parameter_file_name = "parameter_{}.cfg".format(file_index)
        parameter_file_name = os.path.join(self.working_dir, parameter_file_name)
        with open(parameter_file_name, "w") as parameter_file:
            nni_log(LogType.Info, '%s: saving parameter %s' % (self.name, parameters["value"]))
            parameter_file.write(parameters["value"])

    def is_running(self):
        if (self.process is None):
            return False

        retCode = self.process.poll()
        # child worker process exits and all stdout data is read
        if retCode is not None and self.log_pipe_stdout.set_process_exit() and self.log_pipe_stdout.is_read_completed == True:
            # In Windows, the retCode -1 is 4294967295. It's larger than c_long, and raise OverflowError.
            # So covert it to int32.
            retCode = ctypes.c_long(retCode).value
            nni_log(LogType.Info, '{0}: subprocess terminated. Exit code is {1}.'.format(self.name, retCode))

            end_time = int(datetime.now().timestamp() * 1000)
            end_message = {
                "code": retCode,
                "time": end_time,
                "trial": self.id,
            }
            self.command_channel.send(CommandType.TrialEnd, end_message)
            self.cleanup()
            return False
        else:
            return True

    def kill(self, trial_id=None):
        if trial_id == self.id or trial_id is None:
            if self.process is not None:
                try:
                    nni_log(LogType.Info, "%s: killing trial" % self.name)
                    for child in psutil.Process(self.process.pid).children(True):
                        child.kill()
                    self.process.kill()
                except psutil.NoSuchProcess:
                    nni_log(LogType.Info, "kill trial %s failed: %s does not exist!" % (trial_id, self.process.pid))
                except Exception as ex:
                    nni_log(LogType.Error, "kill trial %s failed: %s " % (trial_id, str(ex)))
            self.cleanup()

    def cleanup(self):
        nni_log(LogType.Info, "%s: clean up trial" % self.name)
        self.process = None
        if self.log_pipe_stdout is not None:
            self.log_pipe_stdout.set_process_exit()
            self.log_pipe_stdout = None
        if self.trial_syslogger_stdout is not None:
            self.trial_syslogger_stdout.close()
            self.trial_syslogger_stdout = None
