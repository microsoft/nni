# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import importlib
import os
from pathlib import Path
import sys
import yaml
from nni.runtime.config import get_config_file

ALGO_TYPES = ['tuners', 'assessors', 'advisors']

def get_all_builtin_names(algo_type):
    """Get all builtin names of registered algorithms of specified type

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

    return [x['builtinName'] for x in read_registerd_algo_meta()[algo_type]]


def get_registered_algo_meta(builtin_name, algo_type=None):
    """ Get meta information of registered algorithms.

    Parameters
    ----------
    builtin_name: str
        builtin name.
    algo_type: str | None
        can be one of 'tuners', 'assessors', 'advisors' or None

    Returns: dict | None
    -------
        Returns meta information of speicified builtin alogorithms, for example:
        {
            'classArgsValidator': 'nni.smac_tuner.SMACClassArgsValidator',
            'className': 'nni.smac_tuner.SMACTuner',
            'builtinName': 'SMAC'
        }
    """
    assert builtin_name is not None
    if algo_type:
        assert algo_type in ALGO_TYPES
    config = read_registerd_algo_meta()

    candidates = []
    if algo_type:
        candidates = config[algo_type]
    else:
        for algo_type in ALGO_TYPES:
            candidates.extend(config[algo_type])
    for meta in candidates:
        if meta['builtinName'] == builtin_name:
            return meta
    return None

def parse_full_class_name(full_class_name):
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
    meta = get_registered_algo_meta(builtin_name, algo_type)
    if not meta:
        return None, None
    return parse_full_class_name(meta['className'])

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
    meta = get_registered_algo_meta(builtin_name, algo_type)
    if not meta or 'classArgsValidator' not in meta:
        return None
    module_name, class_name = parse_full_class_name(meta['classArgsValidator'])
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
        module_name, class_name = parse_full_class_name(algo_meta['className'])

        class_args = {}
        if 'classArgs' in algo_meta:
            class_args = algo_meta['classArgs']
        if input_class_args is not None:
            class_args.update(input_class_args)

        return module_name, class_name, class_args

    algo_meta = get_registered_algo_meta(builtin_name, algo_type)
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
            codeDirectory: code directory
            className: qualified class name
            classArgs (optional): kwargs pass to class constructor

    Returns: object
    -------
        Returns customized class instance.
    """

    code_dir = class_params.get('codeDirectory')
    qualified_class_name = class_params.get('className')
    class_args = class_params.get('classArgs')

    if code_dir and not os.path.isdir(code_dir):
        raise ValueError(f'Directory not found: {code_dir}')

    sys.path.append(code_dir)
    module_name, class_name = qualified_class_name.rsplit('.', 1)
    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, class_name)

    if class_args is None:
        class_args = {}
    instance = class_constructor(**class_args)

    return instance

def _using_conda_or_virtual_environment():
    return sys.prefix != sys.base_prefix or os.path.isdir(os.path.join(sys.prefix, 'conda-meta'))

def get_registered_algo_config_path():
    return str(get_config_file('registered_algorithms.yml'))

def read_registerd_algo_meta():
    config_file = get_registered_algo_config_path()
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = defaultdict(list)
    for t in ALGO_TYPES:
        if t not in config:
            config[t] = []
    return config

def write_registered_algo_meta(config):
    config_file = get_registered_algo_config_path()
    with open(config_file, 'w') as f:
        f.write(yaml.safe_dump(dict(config), default_flow_style=False))
