# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import threading
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue

from .log_utils import LogType, nni_log
from .commands import CommandType

INTERVAL_SECONDS = 0.5


class BaseChannel(ABC):
    def __init__(self, args):
        self.is_keep_parsed = args.node_count > 1
        self.args = args
        self.node_id = self.args.node_id

    @abstractmethod
    def _inner_send(self, message):
        pass

    @abstractmethod
    def _inner_receive(self):
        return []

    @abstractmethod
    def _inner_open(self):
        pass

    @abstractmethod
    def _inner_close(self):
        pass

    def open(self):
        # initialize receive, send threads.
        self.is_running = True
        self.receive_queue = Queue()
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.start()
        self.send_queue = Queue()
        self.send_thread = threading.Thread(target=self._send_loop)
        self.send_thread.start()

        self._inner_open()

        client_info = {
            "isReady": True,
            "runnerId": self.args.runner_id,
            "expId": self.args.exp_id,
        }
        nni_log(LogType.Info, 'Channel: send ready information %s' % client_info)
        self.send(CommandType.Initialized, client_info)

    def close(self):
        self.is_running = False
        try:
            self._inner_close()
        except Exception as err:
            # ignore any error on closing
            print("error on closing channel: %s" % err)

    def send(self, command, data):
        """Send command to Training Service.
        command: CommandType object.
        data: string payload.
        the message is sent synchronized.
        """
        data["node"] = self.node_id
        data = json.dumps(data)
        data = data.encode('utf8')
        message = b'%b%014d%b' % (command.value, len(data), data)
        self.send_queue.put(message)

    def sent(self):
        return self.send_queue.qsize() == 0

    def received(self):
        return self.receive_queue.qsize() > 0

    def receive(self):
        """Receive a command from Training Service.
        Returns a tuple of command (CommandType) and payload (str)
        """
        command = None
        data = None

        try:
            command_content = self.receive_queue.get(False)
            if command_content is not None:
                if (len(command_content) < 16):
                    # invalid header
                    nni_log(LogType.Error, 'incorrect command is found, command must be greater than 16 bytes!')
                    return None, None
                header = command_content[:16]
                command = CommandType(header[:2])
                length = int(header[2:])
                if (len(command_content)-16 != length):
                    nni_log(LogType.Error, 'incorrect command length, length {}, actual data length is {}, header {}.'
                            .format(length, len(command_content)-16, header))
                    return None, None
                data = command_content[16:16+length]
                data = json.loads(data.decode('utf8'))
                if self.node_id is None:
                    nni_log(LogType.Info, 'Received command, header: [%s], data: [%s]' % (header, data))
                else:
                    nni_log(LogType.Info, 'Received command(%s), header: [%s], data: [%s]' % (self.node_id, header, data))
        except Empty:
            # do nothing, if no command received.
            pass
        except Exception as identifier:
            nni_log(LogType.Error, 'meet unhandled exception in base_channel: %s' % identifier)
        return command, data

    def _fetch_message(self, buffer, has_new_line=False):
        messages = []
        while(len(buffer)) >= 16:
            header = buffer[:16]
            length = int(header[2:])

            message_length = length+16
            total_length = message_length
            if has_new_line:
                total_length += 1

            # break, if buffer is too short.
            if len(buffer) < total_length:
                break
            data = buffer[16:message_length]
            if has_new_line and 10 != buffer[total_length-1]:
                nni_log(LogType.Error, 'end of message should be \\n, but got {}'.format(self.in_cache[total_length-1]))
            buffer = buffer[total_length:]
            messages.append(header + data)

        return messages, buffer

    def _receive_loop(self):
        while (self.is_running):
            messages = self._inner_receive()
            if messages is not None:
                for message in messages:
                    self.receive_queue.put(message)
            time.sleep(INTERVAL_SECONDS)

    def _send_loop(self):
        while (self.is_running):
            message = None
            try:
                # no sleep, since it's a block call with INTERVAL_SECONDS second timeout
                message = self.send_queue.get(True, INTERVAL_SECONDS)
            except Empty:
                # do nothing, if no command received.
                pass
            if message is not None:
                self._inner_send(message)
