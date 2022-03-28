# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# To make auto-completion happy, we generate a _nn.py that lists out all the classes.

nn_cache_file_path = Path(__file__).parent / '_nn.py'

cache_valid = False

if nn_cache_file_path.exists():
    from . import _nn  # pylint: disable=no-name-in-module
    # valid only when torch version match
    if _nn._torch_version == torch.__version__:
        cache_valid = True

if not cache_valid:
    _NO_WRAP_CLASSES = [
        # not an nn.Module
        'Parameter',
        'ParameterList',
        'UninitializedBuffer',
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

    code = [
        '# This file is auto-generated to make auto-completion work.',
        '# When pytorch version does not match, it will get automatically updated.',
        '# pylint: skip-file',
        f'_torch_version = "{torch.__version__}"',
        'import torch.nn as nn',
        'from nni.retiarii.serializer import basic_unit',
    ]

    all_names = []

    # Add modules, classes, functions in torch.nn into this module.
    for name, obj in inspect.getmembers(torch.nn):
        if inspect.isclass(obj):
            if name in _NO_WRAP_CLASSES:
                code.append(f'{name} = nn.{name}')
            elif not issubclass(obj, nn.Module):
                # It should never go here
                # We did it to play safe
                warnings.warn(f'{obj} is found to be not a nn.Module, which is unexpected. '
                              'It means your PyTorch version might not be supported.', RuntimeWarning)
                code.append(f'{name} = nn.{name}')
            elif name in _WRAP_WITHOUT_TAG_CLASSES:
                code.append(f'{name} = basic_unit(nn.{name}, basic_unit_tag=False)')
            else:
                code.append(f'{name} = basic_unit(nn.{name})')

            all_names.append(name)

        elif inspect.isfunction(obj) or inspect.ismodule(obj):
            code.append(f'{name} = nn.{name}')  # no modification
            all_names.append(name)

    code.append(f'__all__ = {all_names}')

    with nn_cache_file_path.open('w') as fp:
        fp.write('\n'.join(code))


# Import all modules from generated _nn.py

from . import _nn  # pylint: disable=no-name-in-module
__all__ = _nn.__all__
from ._nn import *  # pylint: disable=import-error, wildcard-import
