# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import threading
from enum import Enum


class CommandType(Enum):
    # in
    Initialize = b'IN'
    RequestTrialJobs = b'GE'
    ReportMetricData = b'ME'
    UpdateSearchSpace = b'SS'
    ImportData = b'FD'
    AddCustomizedTrialJob = b'AD'
    TrialEnd = b'EN'
    Terminate = b'TE'
    Ping = b'PI'

    # out
    Initialized = b'ID'
    NewTrialJob = b'TR'
    SendTrialJobParameter = b'SP'
    NoMoreTrialJobs = b'NO'
    KillTrialJob = b'KI'

_lock = threading.Lock()
try:
    _in_file = open(3, 'rb')
    _out_file = open(4, 'wb')
except OSError:
    _msg = 'IPC pipeline not exists, maybe you are importing tuner/assessor from trial code?'
    logging.getLogger(__name__).warning(_msg)


def send(command, data):
    """Send command to Training Service.
    command: CommandType object.
    data: string payload.
    """
    global _lock
    try:
        _lock.acquire()
        data = data.encode('utf8')
        msg = b'%b%014d%b' % (command.value, len(data), data)
        logging.getLogger(__name__).debug('Sending command, data: [%s]', msg)
        _out_file.write(msg)
        _out_file.flush()
    finally:
        _lock.release()


def receive():
    """Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    """
    header = _in_file.read(16)
    logging.getLogger(__name__).debug('Received command, header: [%s]', header)
    if header is None or len(header) < 16:
        # Pipe EOF encountered
        logging.getLogger(__name__).debug('Pipe EOF encountered')
        return None, None
    length = int(header[2:])
    data = _in_file.read(length)
    command = CommandType(header[:2])
    data = data.decode('utf8')
    logging.getLogger(__name__).debug('Received command, data: [%s]', data)
    return command, data
