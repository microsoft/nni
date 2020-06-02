# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import threading
import time
from datetime import datetime
from enum import Enum

from .log_utils import LogType, nni_log

command_path = "./commands"
runner_command_prefix = "runner_command_"
manager_command_prefix = "manager_command_"


class CommandType(Enum):
    Initialize = b'IN'
    RequestTrialJobs = b'GE'
    ReportMetricData = b'ME'
    UpdateSearchSpace = b'SS'
    ImportData = b'FD'
    AddCustomizedTrialJob = b'AD'
    TrialEnd = b'EN'
    Terminate = b'TE'
    Ping = b'PI'

    Initialized = b'ID'
    NewTrialJob = b'TR'
    SendTrialJobParameter = b'SP'
    NoMoreTrialJobs = b'NO'
    KillTrialJob = b'KI'


def send(command, data):
    """Send command to Training Service.
    command: CommandType object.
    data: string payload.
    """

    if not os.path.exists(command_path):
        os.makedirs(command_path)
    while True:
        file_name = os.join(command_path, "%s%s.txt" % (
            runner_command_prefix, int(datetime.now().timestamp * 1000)))
        if (os.path.exists(file_name)):
            time.sleep(0.01)
            continue
        with open(file_name, "wb") as out_file:
            data = json.dumps(data.encode('utf8'))
            msg = b'%b%014d%b' % (command.value, len(data), data)
            nni_log(LogType.Info, 'Sending command, data: [%s]' % msg)
            out_file.write(msg)
        break


def receive():
    """Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    """
    command = None
    data = None

    try:
        pending_commands = []
        if os.path.exists(command_path):
            command_files = os.listdir(command_path)
            for item in command_files:
                if (item.startswith(manager_command_prefix)):
                    pending_commands.append(item)
            pending_commands.sort()

            if len(pending_commands) > 0:
                for command_file in pending_commands:
                    command_file = os.path.join(command_path, command_file)
                    with open(command_file, "rb") as _in_file:
                        header = _in_file.read(16)
                        nni_log(LogType.Info, 'Received command, header: [%s]' % header)
                        if header is None or len(header) < 16:
                            # invalid header
                            nni_log(LogType.Error, 'incorrect command is found!')
                            return None, None
                        length = int(header[2:])
                        data = _in_file.read(length)
                        command = CommandType(header[:2])
                        data = json.loads(data.decode('utf8'))
                        nni_log(LogType.Info, 'Received command, data: [%s]' % data)
                    os.remove(command_file)
    except Exception as identifier:
        nni_log(LogType.Error, 'meet unhandled exception: %s' % identifier)
    return command, data
