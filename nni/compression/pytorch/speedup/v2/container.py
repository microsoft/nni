# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mask_updater import MaskUpdater
    import torch

class Slot:
    def __init__(self) -> None:
        self.value_0 = None
        self.value_1 = None
        self.value_2 = None
        self.value_3 = None
        self.mask_0 = None
        self.mask_1 = None
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
