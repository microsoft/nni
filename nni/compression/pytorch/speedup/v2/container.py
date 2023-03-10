# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

        self._output_origin = None
        self._output_inplace = None
        self._output_randomize = None
        self._output_grad = None
        self._output_masks = None
        self._param_masks = None
        self.assignment_status = {
            'output_origin': 0,
            'output_inplace': 0,
            'output_randomize': 0,
            'output_grad': 0,
            'output_masks': 0,
            'param_masks': 0,
        }

    @property
    def output_origin(self):
        """
        The original output of a node.
        """
        # assert self.assignment_status['output_origin'] == 1, \
        #     f"NodeInfo error: bad output_origin({self.assignment_status['output_origin']})"
        return self._output_origin

    @property
    def output_inplace(self):
        """
        A clone of the original output, used as the input of successor node to get the orginal output of successor node.
        """
        # assert self.assignment_status['output_inplace'] == 1, \
        #     f"NodeInfo error: bad output_inplace({self.assignment_status['output_inplace']})"
        return self._output_inplace

    @property
    def output_randomize(self):
        """
        A randomize output of the original output, used to direct propagate masks.
        """
        # assert self.assignment_status['output_randomize'] == 1, \
        #     f"NodeInfo error: bad output_randomize({self.assignment_status['output_randomize']})"
        return self._output_randomize

    @property
    def output_grad(self):
        """
        The sum of the gradient given by successor during indirect propagation.
        """
        # assert self.assignment_status['output_grad'] == 1, f"NodeInfo error: bad output_grad({self.assignment_status['output_grad']})"
        return self._output_grad

    @property
    def output_masks(self):
        # assert self.assignment_status['output_masks'] <= 3, f"NodeInfo error: bad output_masks({self.assignment_status['output_masks']})"
        return self._output_masks

    @property
    def param_masks(self):
        # assert self.assignment_status['param_masks'] <= 2, f"NodeInfo error: bad param_masks({self.assignment_status['param_masks']})"
        return self._param_masks

    @output_origin.setter
    def output_origin(self, val: Any):
        self._output_origin = val
        self.assignment_status['output_origin'] += 1

    @output_inplace.setter
    def output_inplace(self, val: Any):
        self._output_inplace = val
        self.assignment_status['output_inplace'] += 1

    @output_randomize.setter
    def output_randomize(self, val: Any):
        self._output_randomize = val
        self.assignment_status['output_randomize'] += 1

    @output_grad.setter
    def output_grad(self, val: Any):
        self._output_grad = val
        self.assignment_status['output_grad'] += 1

    @output_masks.setter
    def output_masks(self, val: Any):
        self._output_masks = val
        self.assignment_status['output_masks'] += 1

    @param_masks.setter
    def param_masks(self, val: Any):
        self._param_masks = val
        self.assignment_status['param_masks'] += 1
