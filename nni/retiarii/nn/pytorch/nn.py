# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import warnings

import torch
import torch.nn as nn

from nni.retiarii.serializer import basic_unit

# init all with nothing
__all__ = []

_NO_WRAP_CLASSES = [
    # not an nn.Module
    'Parameter',
    'ParameterList',
    'UninitializedBuffer'
    'UninitializedParameter',

    # arguments are special    
    'Module',
    'Sequential',

    # utilities
    'Container',
    'DataParallel',
]

_WRAP_WITHOUT_TAG_CLASSES = [
    # special support on graph engine
    'ModuleList',
    'ModuleDict',
]

# Add modules, classes, functions in torch.nn into this module.
for name, object in inspect.getmembers(torch.nn):
    if inspect.isclass(object):
        if name in _NO_WRAP_CLASSES:
            globals()[name] = object
        elif not issubclass(object, nn.Module):
            # It should never go here
            # We did it to play safe
            warnings.warn(f'{object} is found to be not a nn.Module, which is unexpected. '
                          'It means your PyTorch version might not be supported.', RuntimeWarning)
            globals()[name] = object
        elif name in _WRAP_WITHOUT_TAG_CLASSES:
            globals()[name] = basic_unit(object, basic_unit_tag=False)
        else:
            # Actually this condition must be true
            # We comment it to play safe
            if not issubclass(object, nn.Module):
                raise 
            globals()[name] = basic_unit(object)

        __all__.append(name)

    elif inspect.isfunction(object) or inspect.ismodule(object):

        globals()[name] = object  # no modification
        __all__.append(name)
