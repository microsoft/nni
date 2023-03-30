# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import sys
import warnings

import cloudpickle
import json_tricks
import numpy
import yaml

import nni

def _minor_version_tuple(version_str: str) -> tuple[int, int]:
    # If not a number, returns -1 (e.g., 999.dev0 -> (999, -1))
    return tuple(int(x) if x.isdigit() else -1 for x in version_str.split(".")[:2])

PYTHON_VERSION = sys.version_info[:2]
NUMPY_VERSION = _minor_version_tuple(numpy.__version__)

try:
    import torch
    TORCH_VERSION = _minor_version_tuple(torch.__version__)
except ImportError:
    logging.getLogger(__name__).debug("PyTorch is not installed.")
    TORCH_VERSION = None

try:
    import pytorch_lightning
    PYTORCH_LIGHTNING_VERSION = _minor_version_tuple(pytorch_lightning.__version__)
except ImportError:
    logging.getLogger(__name__).debug("PyTorch Lightning is not installed.")
    PYTORCH_LIGHTNING_VERSION = None

try:
    import tensorflow
    TENSORFLOW_VERSION = _minor_version_tuple(tensorflow.__version__)
except ImportError:
    logging.getLogger(__name__).debug("Tensorflow is not installed.")
    TENSORFLOW_VERSION = None

# Serialization version check are needed because they are prone to be inconsistent between versions

CLOUDPICKLE_VERSION = _minor_version_tuple(cloudpickle.__version__)
JSON_TRICKS_VERSION = _minor_version_tuple(json_tricks.__version__)
PYYAML_VERSION = _minor_version_tuple(yaml.__version__)

NNI_VERSION = _minor_version_tuple(nni.__version__)

def version_dump() -> dict[str, tuple[int, int] | None]:
    return {
        'python': PYTHON_VERSION,
        'numpy': NUMPY_VERSION,
        'torch': TORCH_VERSION,
        'pytorch_lightning': PYTORCH_LIGHTNING_VERSION,
        'tensorflow': TENSORFLOW_VERSION,
        'cloudpickle': CLOUDPICKLE_VERSION,
        'json_tricks': JSON_TRICKS_VERSION,
        'pyyaml': PYYAML_VERSION,
        'nni': NNI_VERSION
    }

def version_check(expect: dict, raise_error: bool = False) -> None:
    current_ver = version_dump()
    for package in expect:
        # version could be list due to serialization
        exp_version: tuple | None = tuple(expect[package]) if expect[package] else None
        if exp_version is None:
            continue

        err_message: str | None = None
        if package not in current_ver:
            err_message = f'{package} is missing in current environment'
        elif current_ver[package] != exp_version:
            err_message = f'Expect {package} to have version {exp_version}, but {current_ver[package]} found'
        if err_message:
            if raise_error:
                raise RuntimeError('Version check failed: ' + err_message)
            else:
                warnings.warn('Version check with warning: ' + err_message)


def torch_version_is_2() -> bool:
    if TORCH_VERSION is None:
        return False
    if TORCH_VERSION < (2, 0):
        return False
    else:
        return True
