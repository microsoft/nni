# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
import os
from pathlib import Path
import shutil
import sys

import nni

# the version when default config got modified
# don't need to update if default config does not change
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
        config_dir.mkdir(parents=True)
        info = f'Created by NNI {nni.__version__} at {datetime.now()}'
        (config_dir / f'version-{latest_config_version}').write_text(info)

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
    # nni creates a file like "version-2.6" in config dir to identify the version of config files
    # the number means that config content was lastly updated at version 2.6, while nni's version might be higher

    # version number is saved in file name because it's faster to check

    version_file = config_dir / f'version-{latest_config_version}'
    if version_file.exists():
        return

    old_version = 0
    for file in config_dir.iterdir():
        if str(file).startswith('version-'):
            old_version = str(file)[8:]
            file.unlink()
    info = f'Updated by NNI {nni.__version__} from {old_version} at {datetime.now()}'
    version_file.write_text(info)

    from nni.tools.package_utils import update_algo_config
    default_dir = Path(nni.__path__[0], 'runtime/default_config')
    update_algo_config(config_dir, default_dir)
