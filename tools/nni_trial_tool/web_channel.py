# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio

import websockets

from .base_channel import BaseChannel
from .log_utils import LogType, nni_log


class WebChannel(BaseChannel):

    def __init__(self, args):
        self.node_id = args.node_id
        self.args = args
        self.client = None
        self.in_cache = b""

        super(WebChannel, self).__init__(args)

        self._event_loop = None

    def _inner_open(self):
        url = "ws://{}:{}".format(self.args.nnimanager_ip, self.args.nnimanager_port)
        nni_log(LogType.Info, 'WebChannel: connected with info %s' % url)

        connect = websockets.connect(url)
        self._event_loop = asyncio.get_event_loop()
        client = self._event_loop.run_until_complete(connect)
        self.client = client

    def _inner_close(self):
        if self.client is not None:
            self.client.close()
            if self._event_loop.is_running():
                self._event_loop.close()
            self.client = None
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
