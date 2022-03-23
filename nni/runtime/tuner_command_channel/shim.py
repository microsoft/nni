# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Compatibility layer for old protocol APIs.
"""

from __future__ import annotations

from .command_type import CommandType
from . import web_socket_sync

def send(command_type: CommandType, data: str) -> None:
    command = command_type.value.decode() + data
    web_socket_sync.send(command)

def receive() -> tuple[CommandType, str]:
    command = web_socket_sync.receive()
    if command is None:
        raise RuntimeError('NNI manager closed connection')
    command_type = CommandType(command[:2].encode())
    if command_type is CommandType.Terminate:
        web_socket_sync.disconnect()
    return command_type, command[2:]
