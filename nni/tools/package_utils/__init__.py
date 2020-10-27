# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import importlib
import os
from pathlib import Path
import sys

import ruamel.yaml as yaml

import nni
from .constants import BuiltinAlgorithms

ALGO_TYPES = ['tuners', 'assessors', 'advisors']

def get_all_builtin_names(algo_type):
    """Get all valid builtin names, including:
    1. BuiltinAlgorithms which is pre-installed.
    2. User installed packages in <nni_installation_path>/config/installed_packages.yml

    Parameters
    ----------
    algo_type: str
        can be one of 'tuners', 'assessors' or 'advisors'

    Returns: list of string
    -------
    All builtin names of specified type, for example, if algo_type is 'tuners', returns
    all builtin tuner names.
    """
    assert algo_type in ALGO_TYPES
    merged_dict = _get_merged_builtin_dict()

    builtin_names = [x['name'] for x in merged_dict[algo_type]]
    return builtin_names

def get_not_installable_builtin_names(algo_type=None):
    """Get builtin names in BuiltinAlgorithms which do not need to be installed
    and can be used once NNI is installed.

    Parameters
    ----------
    algo_type: str | None
        can be one of 'tuners', 'assessors', 'advisors' or None

    Returns: list of string
    -------
    All builtin names of specified type, for example, if algo_type is 'tuners', returns
    all builtin tuner names.
    If algo_type is None, returns all builtin names of all types.
    """
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
    """ Get meta information of builtin algorithms from:
    1. Pre-installed BuiltinAlgorithms
    2. User installed packages in <nni_installation_path>/config/installed_packages.yml

    Parameters
    ----------
    algo_type: str | None
        can be one of 'tuners', 'assessors', 'advisors' or None
    builtin_name: str | None
        builtin name.

    Returns: dict | list of dict | None
    -------
        If builtin_name is specified, returns meta information of speicified builtin
        alogorithms, for example:
        {
            'name': 'Random',
            'class_name': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner',
            'class_args': {
                'algorithm_name': 'random_search'
            },
            'accept_class_args': False,
            'class_args_validator': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptClassArgsValidator'
        }
        If builtin_name is None, returns multiple meta information in a list.
    """
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

def get_installed_package_meta(algo_type, builtin_name):
    """ Get meta information of user installed algorithms from:
    <nni_installation_path>/config/installed_packages.yml

    Parameters
    ----------
    algo_type: str | None
        can be one of 'tuners', 'assessors', 'advisors' or None
    builtin_name: str
        builtin name.

    Returns: dict | None
    -------
        Returns meta information of speicified builtin alogorithms, for example:
        {
            'class_args_validator': 'nni.smac_tuner.smac_tuner.SMACClassArgsValidator',
            'class_name': 'nni.smac_tuner.smac_tuner.SMACTuner',
            'name': 'SMAC'
        }
    """
    assert builtin_name is not None
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

def _parse_full_class_name(full_class_name):
    if not full_class_name:
        return None, None
    parts = full_class_name.split('.')
    module_name, class_name = '.'.join(parts[:-1]), parts[-1]
    return module_name, class_name

def get_builtin_module_class_name(algo_type, builtin_name):
    """Get module name and class name of all builtin algorithms

    Parameters
    ----------
    algo_type: str
        can be one of 'tuners', 'assessors', 'advisors'
    builtin_name: str
        builtin name.

    Returns: tuple
    -------
        tuple of (module name, class name)
    """
    assert algo_type in ALGO_TYPES
    assert builtin_name is not None
    meta = get_builtin_algo_meta(algo_type, builtin_name)
    if not meta:
        return None, None
    return _parse_full_class_name(meta['class_name'])

def create_validator_instance(algo_type, builtin_name):
    """Create instance of validator class

    Parameters
    ----------
    algo_type: str
        can be one of 'tuners', 'assessors', 'advisors'
    builtin_name: str
        builtin name.

    Returns: object | None
    -------
        Returns validator class instance.
        If specified validator class does not exist, returns None.
    """
    assert algo_type in ALGO_TYPES
    assert builtin_name is not None
    meta = get_builtin_algo_meta(algo_type, builtin_name)
    if not meta or 'class_args_validator' not in meta:
        return None
    module_name, class_name = _parse_full_class_name(meta['class_args_validator'])
    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, class_name)

    return class_constructor()

def create_builtin_class_instance(builtin_name, input_class_args, algo_type):
    """Create instance of builtin algorithms

    Parameters
    ----------
    builtin_name: str
        builtin name.
    input_class_args: dict
        kwargs for builtin class constructor
    algo_type: str
        can be one of 'tuners', 'assessors', 'advisors'

    Returns: object
    -------
        Returns builtin class instance.
    """
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
    """Create instance of customized algorithms

    Parameters
    ----------
    class_params: dict
        class_params should contains following keys:
            codeDir: code directory
            classFileName: python file name of the class
            className: class name
            classArgs (optional): kwargs pass to class constructor
    Returns: object
    -------
        Returns customized class instance.
    """

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

def get_package_config_path():
    # FIXME: this might not be the desired location
    config_dir = Path(nni.__path__[0]).parent / 'nni_config'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, 'installed_packages.yml')

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
