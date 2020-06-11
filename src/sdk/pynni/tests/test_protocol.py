# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import nni.protocol
from nni.protocol import CommandType, send, receive

from io import BytesIO
from unittest import TestCase, main


def _prepare_send():
    nni.protocol._out_file = BytesIO()
    return nni.protocol._out_file

def _prepare_receive(data):
    nni.protocol._in_file = BytesIO(data)


class ProtocolTestCase(TestCase):
    def test_send_en(self):
        out_file = _prepare_send()
        send(CommandType.NewTrialJob, 'CONTENT')
        self.assertEqual(out_file.getvalue(), b'TR00000000000007CONTENT')

    def test_send_zh(self):
        out_file = _prepare_send()
        send(CommandType.NewTrialJob, '你好')
        self.assertEqual(out_file.getvalue(), 'TR00000000000006你好'.encode('utf8'))

    def test_receive_en(self):
        _prepare_receive(b'IN00000000000005hello')
        command, data = receive()
        self.assertIs(command, CommandType.Initialize)
        self.assertEqual(data, 'hello')

    def test_receive_zh(self):
        _prepare_receive('IN00000000000006世界'.encode('utf8'))
        command, data = receive()
        self.assertIs(command, CommandType.Initialize)
        self.assertEqual(data, '世界')


if __name__ == '__main__':
    main()
