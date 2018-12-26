# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


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
        self.assertEqual(out_file.getvalue(), b'TR000007CONTENT')

    def test_send_zh(self):
        out_file = _prepare_send()
        send(CommandType.NewTrialJob, '你好')
        self.assertEqual(out_file.getvalue(), 'TR000006你好'.encode('utf8'))

    def test_send_too_large(self):
        _prepare_send()
        exception = None
        try:
            send(CommandType.NewTrialJob, ' ' * 1000000)
        except AssertionError as e:
            exception = e
        self.assertIsNotNone(exception)

    def test_receive_en(self):
        _prepare_receive(b'IN000005hello')
        command, data = receive()
        self.assertIs(command, CommandType.Initialize)
        self.assertEqual(data, 'hello')

    def test_receive_zh(self):
        _prepare_receive('IN000006世界'.encode('utf8'))
        command, data = receive()
        self.assertIs(command, CommandType.Initialize)
        self.assertEqual(data, '世界')


if __name__ == '__main__':
    main()
