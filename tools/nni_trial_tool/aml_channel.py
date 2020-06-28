# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import websockets

from azureml.core.run import Run
from .base_channel import BaseChannel
from .log_utils import LogType, nni_log


class AMLChannel(BaseChannel):
    def __init__(self, args):
        self.args = args
        self.run = Run.get_context()
        super(AMLChannel, self).__init__(args)
        self.current_message_index = -1

    def _inner_open(self):
        pass

    def _inner_close(self):
        pass

    def _inner_send(self, message):
        self.run.log('trial_runner', str(message))

    def _inner_receive(self):
        messages = []
        # receive message is string, to get consistent result, encode it here.
        message_dict = self.run.get_metrics()
        message_list = message_dict['nni_manager']
        if not message_list:
            return messages
        if type(message_list) is list:
            if self.current_message_index < len(message_list) - 1:
                messages = message_list[self.current_message_index + 1 : len(message_list)]
                self.current_message_index = len(message_list) - 1
        elif self.current_message_index == -1:
            messages = [message_list] 
            self.current_message_index += 1
        return messages
