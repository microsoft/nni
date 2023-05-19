# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = [
    'CommandType',
    'LegacyCommandChannel',
    'send',
    'receive',
    '_set_in_file',
    '_set_out_file',
    '_get_out_file',
]

import logging
import os
import threading

from .command_type import CommandType

_logger = logging.getLogger(__name__)


_lock = threading.Lock()
try:
    if os.environ.get('NNI_PLATFORM') != 'unittest':
        _in_file = open(3, 'rb')
        _out_file = open(4, 'wb')
except OSError:
    _logger.debug('IPC pipeline not exists')

def _set_in_file(in_file):
    global _in_file
    _in_file = in_file

def _set_out_file(out_file):
    global _out_file
    _out_file = out_file

def _get_out_file():
    return _out_file

class LegacyCommandChannel:
    def connect(self):
        pass

    def disconnect(self):
        pass

    def _send(self, command, data):
        send(command, data)

    def _receive(self):
        return receive()

def send(command, data):
    """Send command to Training Service.
    command: CommandType object.
    data: string payload.
    """
    global _lock
    try:
        _lock.acquire()
        data = data.encode('utf8')
        msg = b'%b%014d%b' % (command.value.encode(), len(data), data)
        _logger.debug('Sending command, data: [%s]', msg)
        _out_file.write(msg)
        _out_file.flush()
    finally:
        _lock.release()


def receive():
    """Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    """
    header = _in_file.read(16)
    _logger.debug('Received command, header: [%s]', header)
    if header is None or len(header) < 16:
        # Pipe EOF encountered
        _logger.debug('Pipe EOF encountered')
        return None, None
    length = int(header[2:])
    data = _in_file.read(length)
    command = CommandType(header[:2].decode())
    data = data.decode('utf8')
    _logger.debug('Received command, data: [%s]', data)
    return command, data
