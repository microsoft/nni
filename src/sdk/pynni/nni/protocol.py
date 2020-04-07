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

    _Continue = b'_C'
    _End = b'_E'

_lock = threading.Lock()
try:
    _in_file = open(3, 'rb')
    _out_file = open(4, 'wb')
except OSError:
    _msg = 'IPC pipeline not exists, maybe you are importing tuner/assessor from trial code?'
    logging.getLogger(__name__).warning(_msg)

_max_msg_len = 999999

def send(command, data):
    """Send command to Training Service.
    command: CommandType object.
    data: string payload.
    """
    global _lock
    try:
        _lock.acquire()
        data = data.encode('utf8')
        logging.getLogger(__name__).debug('Sending command %s, data: [%s]', command.value, data)
        if len(data) <= _max_msg_len:
            msg = b'%b%06d%b' % (command.value, len(data), data)
        else:
            msg = b'%b______%b' % (command.value, data[:_max_msg_len])
            data = data[_max_msg_len:]
            while len(data) > _max_msg_len:
                msg += b'_C999999%b' % data[:_max_msg_len]
                data = data[_max_msg_len:]
            msg += b'_E%06d%b' % (len(data), data)
        _out_file.write(msg)
        _out_file.flush()
    finally:
        _lock.release()


def receive():
    """Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    """
    header = _in_file.read(8)
    logging.getLogger(__name__).debug('Received command, header: [%s]', header)
    if header is None or len(header) < 8:
        # Pipe EOF encountered
        logging.getLogger(__name__).debug('Pipe EOF encountered')
        return None, None
    command = CommandType(header[:2])
    length = header[2:]
    if length == b'______':
        data = _in_file.read(_max_msg_len)
        while True:
            ctrl_cmd = _in_file.read(2)
            length = _in_file.read(6)
            data += _in_file.read(int(length))
            if ctrl_cmd == '_E':
                break
            elif ctrl_cmd != '_C':
                raise RuntimeError('Unexpected splitted command header %s', ctrl_cmd)
    else:
        data = _in_file.read(int(length))
    data = data.decode('utf8')
    logging.getLogger(__name__).debug('Received command, data: [%s]', data)
    return command, data
