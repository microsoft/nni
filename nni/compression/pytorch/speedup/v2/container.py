from typing import Union

import torch
from torch import nn

class Slot:
    def __init__(self) -> None:
        self.value_0 = None
        self.value_1 = None
        self.value_2 = None
        self.mask_1 = None
        self.mask_2 = None
        self.status = {
            'value_0': 0,
            'value_1': 0,
            'value_2': 0,
            'mask_1': 0,
            'mask_2': 0,
        }

class NodeInfo:
    def __init__(self, param_masks: dict[str, torch.Tensor]) -> None:
        self.param_masks_0 = param_masks
        self.param_masks_1 = {}
        self.status = {
            'param_0': 0,
            'param_1': 0,
        }
