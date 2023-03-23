# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import time

import nni
from ..base import Command, CommandChannel
from .connection import WsConnection

_logger = logging.getLogger(__name__)

class WsChannelClient(CommandChannel):
    def __init__(self, url: str):
        self._url: str = url
        self._closing: bool = False
        self._conn: WsConnection | None = None

    def connect(self) -> None:
        _logger.debug(f'Connect to {self._url}')
        assert not self._closing
        self._ensure_conn()

    def disconnect(self) -> None:
        _logger.debug(f'Disconnect from {self._url}')
        if self._closing:
            _logger.debug('Already closing')
        else:
            try:
                self._conn.send({'type': '_bye_'})
            except Exception as e:
                _logger.debug(f'Failed to send bye: {repr(e)}')
            self._closing = True
            self._close_conn('client intentionally close')

    def send(self, command: Command) -> None:
        if self._closing:
            return
        _logger.debug(f'Send {command}')
        msg = nni.dump(command)
        for i in range(5):
            try:
                conn = self._ensure_conn()
                conn.send(msg)
                return
            except Exception:
                _logger.exception(f'Failed to send command. Retry in {i}s')
                self._terminate_conn('send fail')
                time.sleep(i)
        _logger.warning(f'Failed to send command {command}. Last retry')
        conn = self._ensure_conn()
        conn.send(msg)

    def receive(self) -> Command | None:
        while True:
            if self._closing:
                return None
            msg = self._receive_msg()
            if msg is None:
                return None
            command = nni.load(msg)
            if command['type'] == '_nop_':
                continue
            if command['type'] == '_bye_':
                reason = command.get('reason')
                _logger.debug(f'Server close connection: {reason}')
                self._closing = True
                self._close_conn('server intentionally close')
                return None
            return command

    def _ensure_conn(self) -> WsConnection:
        if self._conn is None and not self._closing:
            self._conn = WsConnection(self._url)
            self._conn.connect()
            _logger.debug('Connected')
        return self._conn  # type: ignore

    def _close_conn(self, reason: str) -> None:
        if self._conn is not None:
            try:
                self._conn.disconnect(reason)
            except Exception:
                pass
            self._conn = None

    def _terminate_conn(self, reason: str) -> None:
        if self._conn is not None:
            try:
                self._conn.terminate(reason)
            except Exception:
                pass
            self._conn = None

    def _receive_msg(self) -> str | None:
        for i in range(5):
            try:
                conn = self._ensure_conn()
                msg = conn.receive()
                _logger.debug(f'Receive {msg}')
                if not self._closing:
                    assert msg is not None
                return msg
            except Exception:
                _logger.exception(f'Failed to receive command. Retry in {i}s')
                self._terminate_conn('receive fail')
                time.sleep(i)
        _logger.warning(f'Failed to receive command. Last retry')
        conn = self._ensure_conn()
        conn.receive()
