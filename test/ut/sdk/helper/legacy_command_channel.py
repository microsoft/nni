# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.runtime.tuner_command_channel.legacy import send, receive

class LegacyCommandChannel:
    def connect(self):
        pass

    def disconnect(self):
        pass

    def _send(self, command_type, data):
        send(command_type, data)

    def _receive(self):
        return receive()
