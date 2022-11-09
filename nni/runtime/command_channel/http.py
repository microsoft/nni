# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests

from .base import Command, CommandChannel

class HttpChannel(CommandChannel):
    """
    A simple command channel based on HTTP.

    Check the server for details. (``ts/nni_manager/common/command_channel/http.ts``)
    """

    def __init__(self, url: str):
        self._url: str = url

    def send(self, command: Command) -> None:
        requests.put(self._url, json=command)

    def receive(self) -> Command:
        while True:
            r = requests.get(self._url)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 408:
                continue
            raise IOError(f'HTTP command channel received unexpected status code {r.status_code}')
