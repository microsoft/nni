# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Low level APIs for algorithms to communicate with NNI manager.
"""

from __future__ import annotations

__all__ = ['TunerCommandChannel']

from .command_type import CommandType
from .websocket import WebSocket

class TunerCommandChannel:
    """
    A channel to communicate with NNI manager.

    Each NNI experiment has a channel URL for tuner/assessor/strategy algorithm.
    The channel can only be connected once, so for each Python side :class:`~nni.experiment.Experiment` object,
    there should be exactly one corresponding ``TunerCommandChannel`` instance.

    :meth:`connect` must be invoked before sending or receiving data.

    The constructor does not have side effect so ``TunerCommandChannel`` can be created anywhere.
    But :meth:`connect` requires an initialized NNI manager, or otherwise the behavior is unpredictable.

    :meth:`_send` and :meth:`_receive` are underscore-prefixed because their signatures are scheduled to change by v3.0.

    Parameters
    ----------
    url
        The command channel URL.
        For now it must be like ``"ws://localhost:8080/tuner"`` or ``"ws://localhost:8080/url-prefix/tuner"``.
    """

    def __init__(self, url: str):
        self._channel = WebSocket(url)

    def connect(self) -> None:
        self._channel.connect()

    def disconnect(self) -> None:
        self._channel.disconnect()

    # TODO: Define semantic command class like `KillTrialJob(trial_id='abc')`.
    # def send(self, command: Command) -> None:
    #     ...
    # def receive(self) -> Command | None:
    #     ...

    def _send(self, command_type: CommandType, data: str) -> None:
        command = command_type.value.decode() + data
        self._channel.send(command)

    def _receive(self) -> tuple[CommandType, str] | tuple[None, None]:
        command = self._channel.receive()
        if command is None:
            raise RuntimeError('NNI manager closed connection')
        command_type = CommandType(command[:2].encode())
        return command_type, command[2:]
