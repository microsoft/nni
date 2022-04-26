# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: disable=unused-import

from .tuner_command_channel.command_type import CommandType
from .tuner_command_channel.legacy import send, receive

# for unit test compatibility
def _set_in_file(in_file):
    from .tuner_command_channel import legacy
    legacy._in_file = in_file

def _set_out_file(out_file):
    from .tuner_command_channel import legacy
    legacy._out_file = out_file

def _get_out_file():
    from .tuner_command_channel import legacy
    return legacy._out_file
