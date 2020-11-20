import torch
import torch.nn as nn
from typing import (Any, Tuple, List, Optional)

__all__ = ['LayerChoice', 'InputChoice', 'ValueChoice']

class LayerChoice(nn.Module):
    def __init__(self, candidate_ops: List, label: str = None):
        super(LayerChoice, self).__init__()
        self.candidate_ops = candidate_ops
        self.label = label

    def forward(self, x):
        return x

class InputChoice(nn.Module):
    def __init__(self, n_chosen: int = 1, reduction: str = 'sum', label: str = None):
        super(InputChoice, self).__init__()
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.label = label

    def forward(self, candidate_inputs: List[Optional['Tensor']]) -> 'Tensor':
        # fake return
        return torch.tensor(candidate_inputs)

class ValueChoice:
    """
    The instance of this class can only be used as input argument,
    when instantiating a pytorch module.
    TODO: can also be used in training approach
    """
    def __init__(self, candidate_values: List[Any]):
        self.candidate_values = candidate_values
