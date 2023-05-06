# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A WebSocket server runs on random port, accepting one single client.

It prints each message received from client to stdout,
and send each line read from stdin to the client.
"""

import asyncio
import sys

import aioconsole
import websockets

sys.stdin.reconfigure(encoding='utf_8')
sys.stdout.reconfigure(encoding='utf_8')
sys.stderr.reconfigure(encoding='utf_8')

_ws = None

async def main():
    await asyncio.gather(
        ws_server(),
        read_stdin()
    )

async def read_stdin():
    async_stdin, _ = await aioconsole.get_standard_streams()
    async for line in async_stdin:
        line = line.decode().strip()
        _debug(f'read from stdin: {line}')
        if line == '_close_':
            break
        await _ws.send(line)

async def ws_server():
    async with websockets.serve(on_connect, 'localhost', 0) as server:
        port = server.sockets[0].getsockname()[1]
        print(port, flush=True)
        _debug(f'port: {port}')
        await asyncio.Future()

async def on_connect(ws):
    global _ws
    _debug('connected')
    _ws = ws
    try:
        async for msg in ws:
            _debug(f'received from websocket: {msg}')
            print(msg, flush=True)
    except websockets.exceptions.ConnectionClosedError:
        pass

def _debug(msg):
    #sys.stderr.write(f'[server-debug] {msg}\n')
    pass

if __name__ == '__main__':
    asyncio.run(main())
