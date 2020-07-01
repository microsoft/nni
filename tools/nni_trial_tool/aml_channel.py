# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from azureml.core.run import Run # pylint: disable=import-error
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
        try:
            self.run.log('trial_runner', message.decode('utf8'))
        except Exception as exception:
            nni_log(LogType.Error, 'meet unhandled exception when send message: %s' % exception)

    def _inner_receive(self):
        messages = []
        message_dict = self.run.get_metrics()
        if 'nni_manager' not in message_dict:
            return []
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
        newMessage = []
        for message in messages:
            # receive message is string, to get consistent result, encode it here.
            newMessage.append(message.encode('utf8'))
        return newMessage
