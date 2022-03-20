# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Compatibility layer for old protocol APIs.
"""

from __future__ import annotations

from .command_type import CommandType
from . import web_socket_channel

def send(command_type: CommandType, data: str) -> None:
    command = command_type.value.decode() + data
    web_socket_channel.send_command(command)

def receive() -> tuple[CommandType, str]:
    command = web_socket_channel.receive_command()
    if command is None:
        raise RuntimeError('NNI manager closed connection')
    command_type = CommandType(command[:2].encode())
    if command_type is CommandType.Terminate:
        web_socket_channel.shutdown()
    return command_type, command[2:]
