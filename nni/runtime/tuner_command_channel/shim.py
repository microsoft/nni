# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Compatibility layer for old protocol APIs.

We are working on more semantic new APIs.
"""

from __future__ import annotations

from .command_type import CommandType
from .websocket import WebSocket

_ws: WebSocket = None  # type: ignore

def connect(url: str) -> None:
    global _ws
    _ws = WebSocket(url)
    _ws.connect()

def send(command_type: CommandType, data: str) -> None:
    command = command_type.value.decode() + data
    _ws.send(command)

def receive() -> tuple[CommandType, str]:
    command = _ws.receive()
    if command is None:
        raise RuntimeError('NNI manager closed connection')
    command_type = CommandType(command[:2].encode())
    if command_type is CommandType.Terminate:
        _ws.disconnect()
    return command_type, command[2:]
