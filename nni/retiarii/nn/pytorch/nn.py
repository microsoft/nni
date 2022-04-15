# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

# To make auto-completion happy, we generate a _nn.py that lists out all the classes.
nn_cache_file_path = Path(__file__).parent / '_nn.py'

# Update this when cache format changes, to enforce an update.
cache_version = 1

def validate_cache() -> bool:
    import torch

    cache_valid = True

    if nn_cache_file_path.exists():
        from . import _nn  # pylint: disable=no-name-in-module
        if _nn._torch_version != torch.__version__:
            cache_valid = False
        elif getattr(_nn, '_torch_nn_cache_version', -1) != cache_version:
            cache_valid = False

    return cache_valid


def write_cache() -> None:
    import inspect
    import warnings

    import torch
    import torch.nn as nn

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
        '# pyright: reportGeneralTypeIssues=false',
        f'_torch_version = "{torch.__version__}"',
        f'_torch_nn_cache_version = {cache_version}',
        'import typing',
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
                code.append(f'{name} = typing.cast(typing.Type[nn.{name}], basic_unit(nn.{name}, basic_unit_tag=False))')
            else:
                code.append(f'{name} = typing.cast(typing.Type[nn.{name}], basic_unit(nn.{name}))')

            all_names.append(name)

        elif inspect.isfunction(obj) or inspect.ismodule(obj):
            code.append(f'{name} = nn.{name}')  # no modification
            all_names.append(name)

    code.append(f'__all__ = {all_names}')

    with nn_cache_file_path.open('w') as fp:
        fp.write('\n'.join(code))

if not validate_cache():
    write_cache()

del Path, validate_cache, write_cache, nn_cache_file_path

# Import all modules from generated _nn.py

from . import _nn  # pylint: disable=no-name-in-module
from ._nn import *  # pylint: disable=import-error, wildcard-import
