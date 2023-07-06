# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

if TYPE_CHECKING:
    from .mask_updater import MaskUpdater

class NodeInfo:
    """
    Stores the infomation of each intermediate variable.
    The relationship and transmission of information can be seen in the comments of `MaskUpdater`.
    """
    def __init__(self, node: Node):
        self.node = node
        self.module: torch.nn.Module = None
        self.mask_updater: 'MaskUpdater' = None
        self.replaced = False

        self.output_origin = None
        self.output_inplace = None
        self.output_randomize = None
        self.output_grad = None
        self.output_masks = None
        self.param_masks = None
