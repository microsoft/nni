# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mask_updater import MaskUpdater
    import torch

class Slot:
    """
    Stores the infomation of each intermediate variable.
    The relationship and transmission of information can be seen in the comments of `MaskUpdater`.
    """
    def __init__(self) -> None:
        # the original value
        # assigned in `propagate_originally_process`
        self.value_0 = None
        # the value at the first-time assign
        # assigned in `direct_process`
        self.value_1 = None
        # the value at the end(may be unequal to value_1 if there are in-placement ops)
        # assigned in `direct_process`
        self.value_2 = None
        # the grad data of the slot
        # assigned in `indirect_process`
        self.value_3 = None
        # pre-assigned mask. `None` equals to `torch.ones`
        self.mask_0 = None
        # the mask of the slot
        # assigned in `direct_process`
        self.mask_1 = None
        # the mask of the slot
        # assigned in `indirect_process`
        self.mask_2 = None
        self.status = {
            'value_0': 0,
            'value_1': 0,
            'value_2': 0,
            'value_3': 0,
            'mask_1': 0,
            'mask_2': 0,
        }

class NodeInfo:
    def __init__(self) -> None:
        self.mask_updater: 'MaskUpdater' = None
        # these are for call_module node
        self.module: torch.nn.Module = None
        self.param_masks_0: dict[str, torch.Tensor] = None
        self.param_masks_1: dict[str, torch.Tensor] = {}
        self.status = {
            'param_0': 0,
            'param_1': 0,
        }
        self.infos = {}
