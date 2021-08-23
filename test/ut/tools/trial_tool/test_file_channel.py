# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import random
import shutil
import string
import sys
import time
import unittest
from argparse import Namespace
from datetime import datetime

from nni.tools.trial_tool.base_channel import CommandType
from nni.tools.trial_tool.file_channel import (FileChannel, command_path,
                                               manager_commands_file_name)

sys.path.append("..")

runner_file_name = "commands/runner_commands.txt"
manager_file_name = "commands/manager_commands.txt"


class FileChannelTest(unittest.TestCase):

    def setUp(self):
        self.args = Namespace()
        self.args.node_count = 1
        self.args.node_id = None
        if os.path.exists(command_path):
            shutil.rmtree(command_path)

    # FIXME:
    # In the docstring of `BaseChannel.send(self, command, data)`,
    # `data` is "string playload".
    # But in its body it treats `data` as a dict.

    #def test_send(self):
    #    fc = None
    #    try:
    #        fc = FileChannel(self.args)
    #        fc.send(CommandType.ReportGpuInfo, "command1")
    #        fc.send(CommandType.ReportGpuInfo, "command2")

    #        self.check_timeout(2, lambda: os.path.exists(runner_file_name))

    #        self.assertTrue(os.path.exists(runner_file_name))
    #        with open(runner_file_name, "rb") as runner:
    #            lines = runner.readlines()
    #        self.assertListEqual(lines, [b'GI00000000000010"command1"\n', b'GI00000000000010"command2"\n'])
    #    finally:
    #        if fc is not None:
    #            fc.close()

    #def test_send_multi_node(self):
    #    fc1 = None
    #    fc2 = None
    #    try:
    #        runner1_file_name = "commands/runner_commands_1.txt"
    #        self.args.node_id = 1
    #        fc1 = FileChannel(self.args)
    #        fc1.send(CommandType.ReportGpuInfo, "command1")
    #        # wait command have enough time to write before closed.

    #        runner2_file_name = "commands/runner_commands_2.txt"
    #        self.args.node_id = 2
    #        fc2 = FileChannel(self.args)
    #        fc2.send(CommandType.ReportGpuInfo, "command1")

    #        self.check_timeout(2, lambda: os.path.exists(runner1_file_name) and os.path.exists(runner2_file_name))

    #        self.assertTrue(os.path.exists(runner1_file_name))
    #        with open(runner1_file_name, "rb") as runner:
    #            lines1 = runner.readlines()
    #        self.assertTrue(os.path.exists(runner2_file_name))
    #        with open(runner2_file_name, "rb") as runner:
    #            lines2 = runner.readlines()
    #        self.assertListEqual(lines1, [b'GI00000000000010"command1"\n'])
    #        self.assertListEqual(lines2, [b'GI00000000000010"command1"\n'])
    #    finally:
    #        if fc1 is not None:
    #            fc1.close()
    #        if fc2 is not None:
    #            fc2.close()

    # FIXME:
    # `fc.received()` tries to read `BaseChannel.receive_queue`
    # `BaseChannel.receive_queue` is defined in `BaseChannel.open()`
    # `fc.open()` is never invoked.

    #def test_receive(self):
    #    fc = None
    #    manager_file = None
    #    try:
    #        fc = FileChannel(self.args)
    #        message = fc.receive()
    #        self.assertEqual(message, (None, None))

    #        os.mkdir(command_path)
    #        manager_file = open(manager_file_name, "wb")
    #        manager_file.write(b'TR00000000000009"manager"\n')
    #        manager_file.flush()

    #        self.check_timeout(2, lambda: fc.received())
    #        message = fc.receive()
    #        self.assertEqual(message, (CommandType.NewTrialJob, "manager"))

    #        manager_file.write(b'TR00000000000010"manager2"\n')
    #        manager_file.flush()

    #        self.check_timeout(2, lambda: fc.received())
    #        message = fc.receive()
    #        self.assertEqual(message, (CommandType.NewTrialJob, "manager2"))
    #    finally:
    #        if fc is not None:
    #            fc.close()
    #        if manager_file is not None:
    #            manager_file.close()

    def check_timeout(self, timeout, callback):
        interval = 0.01
        start = datetime.now().timestamp()
        count = int(timeout / interval)
        for x in range(count):
            if callback():
                break
            time.sleep(interval)
        print("checked {} times, {:3F} seconds".format(x, datetime.now().timestamp()-start))


if __name__ == '__main__':
    unittest.main()
