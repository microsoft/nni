# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import site
import sys
from pathlib import Path
from collections import defaultdict
import ruamel.yaml as yaml

from .constants import BuiltinAlgorithms

ALGO_TYPES = ['tuners', 'assessors', 'advisors']

def get_all_builtin_names(algo_type):
    assert algo_type in ALGO_TYPES
    merged_dict = _get_merged_builtin_dict()

    return [x['name'] for x in merged_dict[algo_type]]

def get_builtin_algo_meta(algo_type=None, builtin_name=None):
    merged_dict = _get_merged_builtin_dict()

    if algo_type is None:
        return merged_dict

    assert algo_type in ALGO_TYPES

    if not builtin_name:
        return merged_dict[builtin_name]

    for m in merged_dict[algo_type]:
        if m['name'] == builtin_name:
            return m
    return None

def get_python_dir(sitepackages_path):
    if sys.platform == "win32":
        return str(Path(sitepackages_path))
    else:
        return str(Path(sitepackages_path).parents[2])

def get_nni_installation_parent_dir():
    ''' Find nni installation parent directory
    '''
    def try_installation_path_sequentially(*sitepackages):
        '''Try different installation path sequentially util nni is found.
        Return None if nothing is found
        '''
        def _generate_installation_path(sitepackages_path):
            python_dir = get_python_dir(sitepackages_path)
            entry_file = os.path.join(python_dir, 'nni', 'main.js')
            if os.path.isfile(entry_file):
                return python_dir
            return None

        for sitepackage in sitepackages:
            python_dir = _generate_installation_path(sitepackage)
            if python_dir:
                return python_dir
        return None

    if os.getenv('VIRTUAL_ENV'):
        # if 'virtualenv' package is used, `site` has not attr getsitepackages, so we will instead use VIRTUAL_ENV
        # Note that conda venv will not have VIRTUAL_ENV
        python_dir = os.getenv('VIRTUAL_ENV')
    else:
        python_sitepackage = site.getsitepackages()[0]
        # If system-wide python is used, we will give priority to using `local sitepackage`--"usersitepackages()" given
        # that nni exists there
        if python_sitepackage.startswith('/usr') or python_sitepackage.startswith('/Library'):
            python_dir = try_installation_path_sequentially(site.getusersitepackages(), site.getsitepackages()[0])
        else:
            python_dir = try_installation_path_sequentially(site.getsitepackages()[0], site.getusersitepackages())

    return python_dir

def get_nni_installation_path():
    ''' Find nni installation directory
    '''
    parent_dir = get_nni_installation_parent_dir()
    if parent_dir:
        entry_file = os.path.join(parent_dir, 'nni', 'main.js')
        if os.path.isfile(entry_file):
            return os.path.join(parent_dir, 'nni')
    return None

def get_nni_config_dir():
    return os.path.join(get_nni_installation_path(), 'config')

def get_package_config_path():
    config_dir = get_nni_config_dir()
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, 'installed_packages.yml')

def read_installed_package_meta():
    config_file = get_package_config_path()
    print(config_file)
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        config = defaultdict(list)
    for t in ALGO_TYPES:
        if t not in config:
            config[t] = []
    return config

def write_package_meta(config):
    config_file = get_package_config_path()
    with open(config_file, 'w') as f:
        f.write(yaml.dump(dict(config), default_flow_style=False))

def _get_merged_builtin_dict():
    def merge_meta_dict(d1, d2):
        res = defaultdict(list)
        for t in ALGO_TYPES:
            res[t] = d1[t] + d2[t]
        return res

    return merge_meta_dict(BuiltinAlgorithms, read_installed_package_meta())
