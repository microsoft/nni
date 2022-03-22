# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import typing

if typing.TYPE_CHECKING or sys.version_info >= (3, 8):
    Literal = typing.Literal
else:
    Literal = typing.Any
