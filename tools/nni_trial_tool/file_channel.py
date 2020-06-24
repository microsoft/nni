# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from .base_channel import BaseChannel

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

    def _inner_open(self):
        pass

    def _inner_close(self):
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
        full_name = os.path.join(command_path, manager_commands_file_name)

        if self.in_file is not None and self.in_file.closed:
            self.in_file = None

        if self.in_file is None and os.path.exists(full_name):
            self.in_file = open(full_name, "rb")
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
                messages, self.in_cache = self._fetch_message(self.in_cache, True)
        return messages
