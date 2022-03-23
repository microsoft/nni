# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
WIP: There will be a `web_socket_channel.py` dealing with command encoding.

To make the migration smooth, use `shim.py` for now.
"""

from .web_socket_sync import connect, disconnect, send, receive
