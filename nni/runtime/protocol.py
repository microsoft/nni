# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .tuner_command_channel.command_type import CommandType
from .tuner_command_channel.legacy import send, receive

# for unit test compatibility
try:
    from .tuner_command_channel.legacy import _in_file, _out_file
except Exception:
    pass
