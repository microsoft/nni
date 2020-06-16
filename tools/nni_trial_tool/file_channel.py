# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from datetime import datetime

from .base_channel import BaseChannel
from .log_utils import LogType, nni_log

command_path = "./commands"
runner_command_prefix = "runner_command_"
manager_command_prefix = "manager_command_"

class FileChannel(BaseChannel):

    def __init__(self, args):
        super(FileChannel, self).__init__(args)
        self.parsed_commands = set()

    def _inner_send(self, message):
        if not os.path.exists(command_path):
            os.makedirs(command_path, exist_ok=True)
        while True:
            file_name = os.path.join(command_path, "%s%s.txt" % (
                runner_command_prefix, int(datetime.now().timestamp() * 1000)))
            if not os.path.exists(file_name):
                break
            time.sleep(0.01)
        with open(file_name, "wb") as out_file:
            out_file.write(message)

    def _inner_receive(self):
        messages = []

        pending_commands = []
        if os.path.exists(command_path):
            command_files = os.listdir(command_path)
            for file_name in command_files:
                if (file_name.startswith(manager_command_prefix)) and file_name not in self.parsed_commands:
                    pending_commands.append(file_name)
            pending_commands.sort()

            for file_name in pending_commands:
                full_file_name = os.path.join(command_path, file_name)
                with open(full_file_name, "rb") as in_file:
                    header = in_file.read(16)
                    if header is None or len(header) < 16:
                        # invalid header
                        nni_log(LogType.Error, 'incorrect command is found!')
                        return None
                    length = int(header[2:])
                    data = in_file.read(length)
                    messages.append(header + data)
                if not self.is_keep_parsed:
                    os.remove(full_file_name)
                self.parsed_commands.add(file_name)
        return messages
