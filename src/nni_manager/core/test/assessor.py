# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

_in_file = open(3, 'rb')
_out_file = open(4, 'wb')


def send(command, data):
    command = command.encode('utf8')
    data = data.encode('utf8')
    msg = b'%b%14d%b' % (command, len(data), data)
    _out_file.write(msg)
    _out_file.flush()


def receive():
    header = _in_file.read(16)
    l = int(header[2:])
    command = header[:2].decode('utf8')
    data = _in_file.read(l).decode('utf8')
    return command, data


print(receive())

send('KI', '')

print(receive())

send('KI', 'hello')

send('KI', '世界')
