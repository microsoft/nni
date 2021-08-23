# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
import shutil
import sys

import nni

def get_config_directory() -> Path:
    """
    Get NNI config directory.
    Create it if not exist.
    """
    if sys.prefix != sys.base_prefix or Path(sys.prefix, 'conda-meta').is_dir():
        config_dir = Path(sys.prefix, 'nni')
    elif sys.platform == 'win32':
        config_dir = Path(os.environ['APPDATA'], 'nni')
    else:
        config_dir = Path.home() / '.config/nni'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def get_config_file(name: str) -> Path:
    """
    Get an NNI config file.
    Copy from `nni/runtime/default_config` if not exist.
    """
    config_file = get_config_directory() / name
    if not config_file.exists():
        default = Path(nni.__path__[0], 'runtime/default_config', name)
        shutil.copyfile(default, config_file)
    return config_file
