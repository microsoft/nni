# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: disable=unused-import

from __future__ import annotations

from .tuner_command_channel.command_type import CommandType
from .tuner_command_channel import legacy

def send(command: CommandType, data: str) -> None:
    legacy.send(command, data)

def receive() -> tuple[CommandType, str] | tuple[None, None]:
    return legacy.receive()

# for unit test compatibility
def _set_in_file(in_file):
    legacy._in_file = in_file

def _set_out_file(out_file):
    legacy._out_file = out_file

def _get_out_file():
    return legacy._out_file
