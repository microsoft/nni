# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Low level APIs for algorithms to communicate with NNI manager.
"""

from __future__ import annotations

__all__ = ['TunerCommandChannel']

import logging
import time

from .command_type import CommandType
from .websocket import WebSocket

_logger = logging.getLogger(__name__)

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
        self._url = url
        self._channel = WebSocket(url)
        self._retry_intervals = [0, 1, 10]

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
        try:
            self._channel.send(command)
        except WebSocket.ConnectionClosed:
            self._retry_send(command)

    def _retry_send(self, command: str) -> None:
        _logger.warning('Connection lost. Trying to reconnect...')
        for i, interval in enumerate(self._retry_intervals):
            _logger.info(f'Attempt #{i}, wait {interval} seconds...')
            time.sleep(interval)
            self._channel = WebSocket(self._url)
            try:
                self._channel.send(command)
                _logger.info('Reconnected.')
                return
            except Exception as e:
                _logger.exception(e)
        _logger.error('Failed to reconnect.')
        raise RuntimeError('Connection lost')

    def _receive(self) -> tuple[CommandType, str] | tuple[None, None]:
        try:
            command = self._channel.receive()
        except WebSocket.ConnectionClosed:
            # this is for robustness and should never happen
            _logger.warning('ConnectionClosed exception on receiving.')
            command = None
        if command is None:
            command = self._retry_receive()
        command_type = CommandType(command[:2].encode())
        return command_type, command[2:]

    def _retry_receive(self) -> str:
        _logger.warning('Connection lost. Trying to reconnect...')
        for i, interval in enumerate(self._retry_intervals):
            _logger.info(f'Attempt #{i}, wait {interval} seconds...')
            time.sleep(interval)
            self._channel = WebSocket(self._url)
            try:
                command = self._channel.receive()
            except WebSocket.ConnectionClosed:
                command = None  # for robustness
            if command is not None:
                _logger.info('Reconnected')
                return command
        _logger.error('Failed to reconnect.')
        raise RuntimeError('Connection lost')
