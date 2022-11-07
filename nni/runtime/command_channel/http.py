# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests

class HttpCommandChannel:
    def __init__(self, url):
        self._url = url

    def send(self, command):
        requests.put(self._url, json=command)

    def receive(self):
        while True:
            r = requests.get(self._url)
            if r.status_code == 200:
                print(r.json())
                return r.json()
            if r.status_code == 408:
                continue
            r.raise_for_status()
            assert False
