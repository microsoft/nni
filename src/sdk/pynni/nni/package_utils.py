# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import site
import sys
from collections import defaultdict
from pathlib import Path
import importlib
import ruamel.yaml as yaml

from .constants import BuiltinAlgorithms

ALGO_TYPES = ['tuners', 'assessors', 'advisors']

def get_all_builtin_names(algo_type):
    assert algo_type in ALGO_TYPES
    merged_dict = _get_merged_builtin_dict()

    builtin_names = [x['name'] for x in merged_dict[algo_type]]
    return builtin_names

def get_not_installable_builtin_names(algo_type=None):
    if algo_type is None:
        meta = BuiltinAlgorithms
    else:
        assert algo_type in ALGO_TYPES
        meta = {
            algo_type: BuiltinAlgorithms[algo_type]
        }
    names = []
    for t in ALGO_TYPES:
        if t in meta:
            names.extend([x['name'] for x in meta[t]])
    return names

def get_builtin_algo_meta(algo_type=None, builtin_name=None):
    merged_dict = _get_merged_builtin_dict()

    if algo_type is None and builtin_name is None:
        return merged_dict

    if algo_type:
        assert algo_type in ALGO_TYPES
        metas = merged_dict[algo_type]
    else:
        metas = merged_dict['tuners'] + merged_dict['assessors'] + merged_dict['advisors']
    if builtin_name:
        for m in metas:
            if m['name'] == builtin_name:
                return m
    else:
        return metas

    return None

def _parse_full_class_name(full_class_name):
    if not full_class_name:
        return None, None
    parts = full_class_name.split('.')
    module_name, class_name = '.'.join(parts[:-1]), parts[-1]
    return module_name, class_name

def get_builtin_module_class_name(algo_type, builtin_name):
    assert algo_type in ALGO_TYPES
    meta = get_builtin_algo_meta(algo_type, builtin_name)
    if not meta:
        return None, None
    return _parse_full_class_name(meta['class_name'])

def create_validator_instance(algo_type, builtin_name):
    assert algo_type in ALGO_TYPES
    meta = get_builtin_algo_meta(algo_type, builtin_name)
    if not meta or 'class_args_validator' not in meta:
        return None
    module_name, class_name = _parse_full_class_name(meta['class_args_validator'])
    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, class_name)

    return class_constructor()

def create_builtin_class_instance(builtin_name, input_class_args, algo_type):
    assert algo_type in ALGO_TYPES
    if builtin_name not in get_all_builtin_names(algo_type):
        raise RuntimeError('Builtin name is not found: {}'.format(builtin_name))

    def parse_algo_meta(algo_meta, input_class_args):
        """
        1. parse class_name field in meta data into module name and class name,
        for example:
            parse class_name 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner' in meta data into:
            module name: nni.hyperopt_tuner.hyperopt_tuner
            class name: HyperoptTuner
        2. merge user specified class args together with builtin class args.
        """
        assert algo_meta
        module_name, class_name = _parse_full_class_name(algo_meta['class_name'])

        class_args = {}
        if 'class_args' in algo_meta:
            class_args = algo_meta['class_args']
        if input_class_args is not None:
            class_args.update(input_class_args)

        return module_name, class_name, class_args

    algo_meta = get_builtin_algo_meta(algo_type, builtin_name)
    module_name, class_name, class_args = parse_algo_meta(algo_meta, input_class_args)

    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError('Builtin module can not be loaded: {}'.format(module_name))

    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, class_name)

    instance = class_constructor(**class_args)

    return instance

def create_customized_class_instance(class_params):
    code_dir = class_params.get('codeDir')
    class_filename = class_params.get('classFileName')
    class_name = class_params.get('className')
    class_args = class_params.get('classArgs')

    if not os.path.isfile(os.path.join(code_dir, class_filename)):
        raise ValueError('Class file not found: {}'.format(
            os.path.join(code_dir, class_filename)))
    sys.path.append(code_dir)
    module_name = os.path.splitext(class_filename)[0]
    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, class_name)

    if class_args is None:
        class_args = {}
    instance = class_constructor(**class_args)

    return instance

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

def get_installed_package_meta(algo_type, builtin_name):
    if algo_type:
        assert algo_type in ALGO_TYPES
    config = read_installed_package_meta()

    candidates = []
    if algo_type:
        candidates = config[algo_type]
    else:
        for algo_type in ALGO_TYPES:
            candidates.extend(config[algo_type])
    for meta in candidates:
        if meta['name'] == builtin_name:
            return meta
    return None

def read_installed_package_meta():
    config_file = get_package_config_path()
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
