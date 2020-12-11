import inspect
import logging
import torch
import torch.nn as nn
from typing import (Any, Tuple, List, Optional)

from ...utils import add_record

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


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

    def forward(self, candidate_inputs: List['Tensor']) -> 'Tensor':
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


class Placeholder(nn.Module):
    def __init__(self, label, related_info):
        add_record(id(self), related_info)
        self.label = label
        self.related_info = related_info
        super(Placeholder, self).__init__()

    def forward(self, x):
        return x


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

class Sequential(nn.Sequential):
    def __init__(self, *args):
        add_record(id(self), {})
        super(Sequential, self).__init__(*args)

class ModuleList(nn.ModuleList):
    def __init__(self, *args):
        add_record(id(self), {})
        super(ModuleList, self).__init__(*args)

def wrap_module(original_class):
    orig_init = original_class.__init__
    argname_list = list(inspect.signature(original_class).parameters.keys())
    # Make copy of original __init__, so we can call it without recursion

    def __init__(self, *args, **kws):
        full_args = {}
        full_args.update(kws)
        for i, arg in enumerate(args):
            full_args[argname_list[i]] = args[i]
        add_record(id(self), full_args)

        orig_init(self, *args, **kws) # Call the original __init__

    original_class.__init__ = __init__ # Set the class' __init__ to the new one
    return original_class

Conv2d = wrap_module(nn.Conv2d)
BatchNorm2d = wrap_module(nn.BatchNorm2d)
ReLU = wrap_module(nn.ReLU)
Dropout = wrap_module(nn.Dropout)
Linear = wrap_module(nn.Linear)
MaxPool2d = wrap_module(nn.MaxPool2d)
AvgPool2d = wrap_module(nn.AvgPool2d)
Identity = wrap_module(nn.Identity)
AdaptiveAvgPool2d = wrap_module(nn.AdaptiveAvgPool2d)
