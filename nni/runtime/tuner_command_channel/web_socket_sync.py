# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for synchronized WebSocket.

This module handles _one_ singleton connection, used by NNI manager <-> tuner/assessor.

According to WebSocket standard, messages are guaranteed not to be fragmented at API level,
so this module does not have buffer like anonymous pipe channel.

FIXME: Need configure websockets library to increase max message length.

TODO: It's better to make "async thread" global.
"""

from __future__ import annotations

import asyncio
import logging
from threading import Thread

import websockets

_logger = logging.getLogger(__name__)

_event_loop = None  # asyncio stdlib is not type-hinted
_ws = None

def connect(url: str) -> None:
    """
    Connect to given URL.
    Start event loop and initialize connection singleton.
    """
    global _event_loop, _ws
    _logger.debug('Connecting...')
    _event_loop = asyncio.new_event_loop()
    Thread(target=_run_event_loop, name='WebSocketEventLoop').start()
    _ws = _wait(_connect_async(url))
    _logger.debug(f'Connected to {url}')

def disconnect() -> None:
    """
    Disconnect and stop event loop.
    Without invoking this function, the event loop will prevent Python interpreter from exiting.
    """
    global _ws
    if _ws is None:
        _logger.debug('disconnect: No connection.')
        return
    try:
        _wait(_ws.close())
        _ws = None
        _logger.debug('Connection closed by client.')
    except Exception as e:
        _logger.warning(f'Failed to close connection: {repr(e)}')
    finally:
        _event_loop.call_soon_threadsafe(_event_loop.stop)

def send(message: str) -> None:
    """
    Send a message.
    """
    _logger.debug(f'Sending {message}')
    _wait(_ws.send(message))

def receive() -> str | None:
    """
    Receive a message, or return ``None`` if the connection has been closed.
    If the connection is closed by server side, this function will stop event loop.
    """
    global _ws
    try:
        msg = _wait(_ws.recv())
        _logger.debug(f'Received {msg}')
    except websockets.ConnectionClosed:
        _logger.debug('Connection closed by server.')
        _ws = None
        _event_loop.call_soon_threadsafe(_event_loop.stop)
        return None

    # seems the library will inference whether it's text or binary, so we don't have guarantee
    if isinstance(msg, bytes):
        return msg.decode()
    else:
        return msg

def _wait(coro):
    # Synchronized version of "await".
    future = asyncio.run_coroutine_threadsafe(coro, _event_loop)
    return future.result()

def _run_event_loop():
    # A separate thread to run the event loop.
    # The event loop itself is blocking, and send/receive are also blocking,
    # so they must run in different threads.
    asyncio.set_event_loop(_event_loop)
    _event_loop.run_forever()
    _logger.debug('Event loop stopped.')

async def _connect_async(url):
    # Theoretically this function is meaningless and one can directly use `websockets.connect(url)`,
    # but it will not work, raising "TypeError: A coroutine object is required".
    # Seems a design flaw in websockets.
    return await websockets.connect(url)
