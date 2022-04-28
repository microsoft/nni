# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: disable=unused-import

from __future__ import annotations

from .tuner_command_channel.command_type import CommandType
from .tuner_command_channel import legacy
from .tuner_command_channel import shim

_use_ws = False

def connect_websocket(url: str):
    global _use_ws
    _use_ws = True
    shim.connect(url)

def send(command: CommandType, data: str) -> None:
    if _use_ws:
        shim.send(command, data)
    else:
        legacy.send(command, data)

def receive() -> tuple[CommandType, str] | tuple[None, None]:
    if _use_ws:
        return shim.receive()
    else:
        return legacy.receive()

# for unit test compatibility
def _set_in_file(in_file):
    legacy._in_file = in_file

def _set_out_file(out_file):
    legacy._out_file = out_file

def _get_out_file():
    return legacy._out_file
