import inspect

import torch.nn as nn

from nni.retiarii import basic_unit

_trace_module_names = [
    module_name for module_name in dir(nn)
    if module_name not in ['Module', 'ModuleList', 'ModuleDict', 'Sequential'] and
    inspect.isclass(getattr(nn, module_name)) and issubclass(getattr(nn, module_name), nn.Module)
]


def remove_inject_pytorch_nn():
    for name in _trace_module_names:
        if hasattr(getattr(nn, name), '__wrapped__'):
            setattr(nn, name, getattr(nn, name).__wrapped__)


def inject_pytorch_nn():
    for name in _trace_module_names:
        setattr(nn, name, basic_unit(getattr(nn, name)))
