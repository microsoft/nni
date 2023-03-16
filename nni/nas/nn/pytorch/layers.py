# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# FIXME: The dynamic generation causes issues upon un-install of NNI or
#        when NNI's been installed at a place with no write access.
#        We should find a better way to do this.

# If you've seen lint errors like `"Sequential" is not a known member of module`,
# please run `python test/vso_tools/trigger_import.py` to generate `_layers.py`.

import hashlib
import os
import warnings
from pathlib import Path

# To make auto-completion happy, we generate a _layers.py that lists out all the classes.
nn_cache_file_path = Path(__file__).parent / '_layers.py'

# Update this when cache format changes, to enforce an update.
cache_version = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:8]


def validate_cache() -> bool:
    import torch

    cache_valid = []

    if nn_cache_file_path.exists():
        lines = nn_cache_file_path.read_text().splitlines()
        for line in lines:
            if line.startswith('# _torch_version = '):
                _cached_torch_version = line[line.find('=') + 1:].strip()
                if _cached_torch_version == torch.__version__:
                    cache_valid.append(True)
            if line.startswith('# _torch_nn_cache_sha1 = '):
                _cached_cache_version = line[line.find('=') + 1:].strip()
                if _cached_cache_version == cache_version:
                    cache_valid.append(True)

    return len(cache_valid) >= 2 and all(cache_valid)


def generate_stub_file() -> str:
    import inspect

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
        '# Copyright (c) Microsoft Corporation.',
        '# Licensed under the MIT license.',
        '# This file is auto-generated to make auto-completion work.',
        '# When pytorch version does not match, it will get automatically updated.',
        '# pylint: skip-file',
        '# pyright: reportGeneralTypeIssues=false',
        f'# _torch_version = {torch.__version__}',
        f'# _torch_nn_cache_version = 10',  # backward compatibility
        f'# _torch_nn_cache_sha1 = {cache_version}',
        'import typing',
        'import torch.nn as nn',
        'from .base import ParametrizedModule',
    ]

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
                # for graph model space
                code.append(f'class {name}(ParametrizedModule, nn.{name}, wraps=nn.{name}, copy_wrapped=True):\n    _nni_basic_unit = False')  # pylint: disable=line-too-long
            else:
                code.append(f'class Mutable{name}(ParametrizedModule, nn.{name}, wraps=nn.{name}): pass')
                # for graph model space
                code.append(f'class {name}(ParametrizedModule, nn.{name}, wraps=nn.{name}, copy_wrapped=True): pass')

        elif inspect.isfunction(obj) or inspect.ismodule(obj):
            code.append(f'{name} = nn.{name}')  # no modification

    return '\n'.join(code)


def write_cache(code: str) -> bool:
    if os.access(nn_cache_file_path.as_posix(), os.W_OK) or (
        not nn_cache_file_path.exists() and os.access(nn_cache_file_path.parent.as_posix(), os.W_OK)
    ):
        with nn_cache_file_path.open('w') as fp:
            fp.write(code)
        return True
    else:
        # no permission
        return False


code = generate_stub_file()

if not validate_cache():
    if not write_cache(code):
        warnings.warn(f'Cannot write to {nn_cache_file_path}. Will try to execute the generated code on-the-fly.')

try:
    # Layers can be either empty or successfully written.
    from ._layers import *  # pylint: disable=import-error, wildcard-import, unused-wildcard-import
except ModuleNotFoundError:
    # Backup plan when the file is not writable.
    exec(code, globals())


def mutable_global_names():
    return [name for name, obj in globals().items() if isinstance(obj, type) and name.startswith('Mutable')]


# Export all the MutableXXX in this module by default.
__all__ = mutable_global_names()  # type: ignore
