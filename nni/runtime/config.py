# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
import shutil
import sys

import nni

latest_config_version = '2.6'

def get_config_directory() -> Path:
    """
    Get NNI config directory.
    Create it if not exist.
    """
    if os.getenv('NNI_CONFIG_DIR') is not None:
        config_dir = Path(os.getenv('NNI_CONFIG_DIR'))
    elif sys.prefix != sys.base_prefix or Path(sys.prefix, 'conda-meta').is_dir():
        config_dir = Path(sys.prefix, 'nni')
    elif sys.platform == 'win32':
        config_dir = Path(os.environ['APPDATA'], 'nni')
    else:
        config_dir = Path.home() / '.config/nni'
    if config_dir.exists():
        _check_and_update_config_version(config_dir)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / f'version_{latest_config_version}').touch()
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

def _check_and_update_config_version(config_dir):
    latest = config_dir / f'version_{latest_config_version}'
    if latest.exists():
        return
    for file in config_dir.iterdir():
        if str(file).startswith('version_'):
            file.unlink()
    latest.touch()

    from nni.tools.package_utils import update_algo_config
    default_dir = Path(nni.__path__[0], 'runtime/default_config')
    update_algo_config(config_dir, default_dir)
