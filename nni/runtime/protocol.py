# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .tuner_command_channel.command_type import CommandType
from .tuner_command_channel import shim

_impl = shim

def send(command_type: CommandType, data: str) -> None:
    _impl.send(command_type, data)

def receive() -> tuple[CommandType, str]:
    return _impl.receive()

def use_legacy_pipe(pipe):
    global _impl
    from .tuner_command_channel import legacy
    legacy._in_file = pipe
    legacy._out_file = pipe
    _impl = legacy
