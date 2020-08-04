# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
import websockets

from .base_channel import BaseChannel
from .log_utils import LogType, nni_log


class WebChannel(BaseChannel):

    def __init__(self, args):
        self.node_id = args.node_id
        self.args = args
        self.client = None
        self.in_cache = b""
        self.timeout = 10

        super(WebChannel, self).__init__(args)

        self._event_loop = None

    def _inner_open(self):
        url = "ws://{}:{}".format(self.args.nnimanager_ip, self.args.nnimanager_port)
        try:
            connect = asyncio.wait_for(websockets.connect(url), self.timeout)
            self._event_loop = asyncio.get_event_loop()
            client = self._event_loop.run_until_complete(connect)
            self.client = client
            nni_log(LogType.Info, 'WebChannel: connected with info %s' % url)
        except asyncio.TimeoutError:
            nni_log(LogType.Error, 'connect to %s timeout! Please make sure NNIManagerIP configured correctly, and accessable.' % url)
            os._exit(1)

    def _inner_close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
            if self._event_loop.is_running():
                self._event_loop.stop()
            self._event_loop = None

    def _inner_send(self, message):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.client.send(message))

    def _inner_receive(self):
        messages = []
        if self.client is not None:
            received = self._event_loop.run_until_complete(self.client.recv())
            # receive message is string, to get consistent result, encode it here.
            self.in_cache += received.encode("utf8")
            messages, self.in_cache = self._fetch_message(self.in_cache)

        return messages
