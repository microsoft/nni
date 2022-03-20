# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
WebSocket IPC channel between NNI manager and tuner/assessor.

The channel is initialized at import time, if environment variable ``NNI_TUNER_COMMAND_CHANNEL`` is set.
The environment variable, if present, should be a WebSocket URL. (e.g. ``ws://localhost:8080/tuner``)

FIXME: Command format should be semantic.
Each command type should have a dataclass and send_command/receive_command should operate on command object.
It will cause API change. Do not use this module directly until it's done.

TODO: Initialization should be controlled by __main__.py. But it's a mess and I don't want to touch.
"""

from __future__ import annotations

import logging
import os

from .command_type import CommandType
from . import web_socket_sync

_logger = logging.getLogger(__name__)

# in future commands will be of `Command` class and this module will deal with encoding/decoding

def send_command(command: str) -> None:
    """
    Send a command to NNI manager.
    """
    web_socket_sync.send(command)

def receive_command() -> str | None:
    """
    Receive a command from NNI manager.

    Returns ``None`` if the connection has been closed.
    Generally this means NNI manager has crashed.
    """
    return web_socket_sync.receive()

def shutdown():
    """
    Shut down the IPC channel.

    If this function is not called, the underlying event loop will prevent Python from exiting.
    """
    web_socket_sync.disconnect()

## initialize ##

if 'NNI_TUNER_COMMAND_CHANNEL' not in os.environ:
    _logger.debug('No NNI_TUNER_COMMAND_CHANNEL environ. This is not tuner process.')
else:
    web_socket_sync.connect(os.environ['NNI_TUNER_COMMAND_CHANNEL'])
