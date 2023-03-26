# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

# except the special case which can not treat as a basic module from pytorch
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
    MutationAnchor = 'MutationAnchor'
    MergedSlice = 'MergedSlice'
    Repeat = 'Repeat'
    Cell = 'Cell'
