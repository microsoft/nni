# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = [
    'get_algo_meta',
    'get_all_algo_meta',
    'register_algo_meta',
    'unregister_algo_meta',
]

from collections import defaultdict
from typing import List, Optional

import yaml

from nni.runtime.config import get_builtin_config_file, get_config_file
from .common import AlgoMeta

def get_algo_meta(name: AlgoMeta) -> Optional[AlgoMeta]:
    """
    Get meta information of a built-in or registered algorithm.
    Return None if not found.
    """
    for algo in get_all_algo_meta():
        if algo.name == name:
            return algo
    return None

def get_all_algo_meta() -> List[AlgoMeta]:
    """
    Get meta information of all built-in and registered algorithms.
    """
    return _load_builtin_config() + _load_custom_config()

def register_algo_meta(algo_meta: AlgoMeta) -> None:
    """
    Register a custom algorithm.
    """
    algos = _load_custom_config()
    algos[algo_meta.name] = algo_meta
    _save_custom_config(algos)

def unregister_algo_meta(algo_name: str) -> None:
    """
    Unregister a custom algorithm.
    """
    algos = _load_custom_config()
    algos.pop(algo_name)
    _save_custom_config(algos)

def _load_builtin_config():
    path = get_builtin_config_file('builtin_algorithms.yml')
    return _load_config_file(path)

def _load_custom_config():
    path = get_config_file('registered_algorithms.yml')
    # for backward compatibility, NNI v2.5- stores all algorithms in this file
    return [algo for algo in  _load_config_file(path) if not algo.is_builtin]

def _load_config_file(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    algos = []
    for algo_type in ['tuner', 'assessor', 'advisor']:
        for algo in config.get(algo_type + 's', []):
            algos.append(AlgoMeta.load(algo, algo_type))
    return algos

def _save_custom_config(algos):
    config = defaultdict(list)
    for algo in algos.values():
        config[algo.algo_type + 's'].append(algo.dump())
    text = yaml.dump(dict(config), default_flow_style=False)
    get_config_file('registered_algorithms.yml').write_text(text)
