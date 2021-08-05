# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

MODULE_EXCEPT_LIST = ['Sequential']


class OpTypeName(str, Enum):
    """
    op type to its type name str
    """
    Attr = 'Attr'
    Constant = 'Constant'
    LayerChoice = 'LayerChoice'
    InputChoice = 'InputChoice'
    ValueChoice = 'ValueChoice'
    Placeholder = 'Placeholder'
    MergedSlice = 'MergedSlice'
    Repeat = 'Repeat'
    Cell = 'Cell'
