# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Synchronized and object-oriented WebSocket class.

WebSocket guarantees that messages will not be divided at API level.
"""

from __future__ import annotations

__all__ = ['WsConnection']

import asyncio
import logging
from threading import Lock, Thread
from typing import Any, Type

import websockets

import nni
from ..base import Command

_logger = logging.getLogger(__name__)

# the singleton event loop
_event_loop: asyncio.AbstractEventLoop = None  # type: ignore
_event_loop_lock: Lock = Lock()
_event_loop_refcnt: int = 0  # number of connected websockets

class WsConnection:
    """
    A WebSocket connection.

    Call :meth:`connect` before :meth:`send` and :meth:`receive`.

    All methods are thread safe.

    Parameters
    ----------
    url
        The WebSocket URL.
        For tuner command channel it should be something like ``ws://localhost:8080/tuner``.
    """

    ConnectionClosed: Type[Exception] = websockets.ConnectionClosed  # type: ignore

    def __init__(self, url: str):
        self._url: str = url
        self._ws: Any = None  # the library does not provide type hints

    def connect(self) -> None:
        global _event_loop, _event_loop_refcnt
        with _event_loop_lock:
            _event_loop_refcnt += 1
            if _event_loop is None:
                _logger.debug('Starting event loop.')
                # following line must be outside _run_event_loop
                # because _wait() might be executed before first line of the child thread
                _event_loop = asyncio.new_event_loop()
                thread = Thread(target=_run_event_loop, name='NNI-WebSocketEventLoop', daemon=True)
                thread.start()

        _logger.debug(f'Connecting to {self._url}')
        self._ws = _wait(_connect_async(self._url))
        _logger.debug(f'Connected.')

    def disconnect(self, reason: str | None = None, code: int | None = None) -> None:
        if self._ws is None:
            _logger.debug('disconnect: No connection.')
            return

        try:
            _wait(self._ws.close(code or 4000, reason))
            _logger.debug('Connection closed by client.')
        except Exception as e:
            _logger.warning(f'Failed to close connection: {repr(e)}')
        self._ws = None
        _decrease_refcnt()

    def terminate(self, reason: str | None = None) -> None:
        if self._ws is None:
            _logger.debug('terminate: No connection.')
            return
        self.disconnect(reason, 4001)

    def send(self, message: Command) -> None:
        _logger.debug(f'Sending {message}')
        try:
            _wait(self._ws.send(nni.dump(message)))
        except websockets.ConnectionClosed:  # type: ignore
            _logger.debug('Connection closed by server.')
            self._ws = None
            _decrease_refcnt()
            raise

    def receive(self) -> Command | None:
        """
        Return received message;
        or return ``None`` if the connection has been closed by peer.
        """
        try:
            msg = _wait(self._ws.recv())
            _logger.debug(f'Received {msg}')
        except websockets.ConnectionClosed:  # type: ignore
            _logger.debug('Connection closed by server.')
            self._ws = None
            _decrease_refcnt()
            raise

        if msg is None:
            return None
        # seems the library will inference whether it's text or binary, so we don't have guarantee
        if isinstance(msg, bytes):
            msg = msg.decode()
        return nni.load(msg)

def _wait(coro):
    # Synchronized version of "await".
    future = asyncio.run_coroutine_threadsafe(coro, _event_loop)
    return future.result()

def _run_event_loop() -> None:
    # A separate thread to run the event loop.
    # The event loop itself is blocking, and send/receive are also blocking,
    # so they must run in different threads.
    asyncio.set_event_loop(_event_loop)
    _event_loop.run_forever()
    _logger.debug('Event loop stopped.')

async def _connect_async(url):
    # Theoretically this function is meaningless and one can directly use `websockets.connect(url)`,
    # but it will not work, raising "TypeError: A coroutine object is required".
    # Seems a design flaw in websockets library.
    return await websockets.connect(url, max_size=None)  # type: ignore

def _decrease_refcnt() -> None:
    global _event_loop, _event_loop_refcnt
    with _event_loop_lock:
        _event_loop_refcnt -= 1
        if _event_loop_refcnt == 0:
            _event_loop.call_soon_threadsafe(_event_loop.stop)
            _event_loop = None  # type: ignore
