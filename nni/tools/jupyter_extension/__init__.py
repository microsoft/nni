# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import proxy

load_jupyter_server_extension = proxy.setup
_load_jupyter_server_extension = proxy.setup
