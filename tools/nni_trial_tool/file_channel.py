# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from datetime import datetime

from .base_channel import BaseChannel
from .log_utils import LogType, nni_log

command_path = "./commands"
runner_commands_file_name_prefix = "runner_commands"
manager_commands_file_name = "manager_commands.txt"


class FileChannel(BaseChannel):

    def __init__(self, args):
        self.node_id = args.node_id
        self.out_file = None
        self.in_file = None
        self.in_offset = 0
        self.in_cache = b""

        super(FileChannel, self).__init__(args)

    def close(self):
        super(FileChannel, self).close()
        if self.out_file is not None:
            self.out_file.close()
            self.out_file = None
        if self.in_file is not None:
            self.in_file.close()
            self.in_file = None

    def _inner_send(self, message):
        if self.out_file is None:
            if not os.path.exists(command_path):
                os.makedirs(command_path, exist_ok=True)

            if self.node_id is None:
                file_name = os.path.join(command_path, "%s.txt" % runner_commands_file_name_prefix)
            else:
                file_name = os.path.join(command_path, "%s_%s.txt" % (
                    runner_commands_file_name_prefix, self.node_id))
            self.out_file = open(file_name, "ab")

        self.out_file.write(message)
        self.out_file.write(b'\n')
        self.out_file.flush()

    def _open_manager_command(self):
        manager_command_file_name = os.path.join(command_path, manager_commands_file_name)

        if self.in_file is not None and self.in_file.closed:
            self.in_file = None

        if self.in_file is None and os.path.exists(manager_command_file_name):
            self.in_file = open(manager_command_file_name, "rb")
            self.in_file.seek(self.in_offset)

    def _inner_receive(self):
        messages = []

        if self.in_file is None:
            self._open_manager_command()
        if self.in_file is not None:
            self.in_file.seek(0, os.SEEK_END)
            new_offset = self.in_file.tell()
            self.in_file.seek(self.in_offset, os.SEEK_SET)
            count = new_offset - self.in_offset
            if count > 0:
                self.in_cache += self.in_file.read(count)
                self.in_offset = new_offset
                while(len(self.in_cache)) >= 16:
                    header = self.in_cache[:16]
                    length = int(header[2:])

                    # consider there is an \n at end of a message.
                    total_length = length+16+1
                    # break, if buffer is too short.
                    if len(self.in_cache) < total_length:
                        break
                    data = self.in_cache[16:total_length-1]
                    if 10 != self.in_cache[total_length-1]:
                        nni_log(LogType.Error, 'end of message should be \\n, but got {}'.format(self.in_cache[total_length-1]))
                    self.in_cache = self.in_cache[total_length:]
                    messages.append(header + data)
        return messages
