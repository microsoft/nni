# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'get_algo_meta',
    'get_all_algo_meta',
    'register_algo_meta',
    'unregister_algo_meta',
]

from collections import defaultdict

import yaml

from nni.runtime.config import get_builtin_config_file, get_config_file
from .common import AlgoMeta

def get_algo_meta(name: str) -> AlgoMeta | None:
    """
    Get meta information of a built-in or registered algorithm.
    Return None if not found.
    """
    name = name.lower()
    for algo in get_all_algo_meta():
        if algo.name.lower() == name:
            return algo
        if algo.alias is not None and algo.alias.lower() == name:
            return algo
    return None

def get_all_algo_meta() -> list[AlgoMeta]:
    """
    Get meta information of all built-in and registered algorithms.
    """
    return _load_builtin_config() + _load_custom_config()

def register_algo_meta(algo_meta: AlgoMeta) -> None:
    """
    Register a custom algorithm.
    If it already exists, overwrite it.
    """
    algos = {algo.name: algo for algo in _load_custom_config()}
    algos[algo_meta.name] = algo_meta
    _save_custom_config(algos.values())

def unregister_algo_meta(algo_name: str) -> None:
    """
    Unregister a custom algorithm.
    If it does not exist, do nothing.
    """
    algos = [algo for algo in _load_custom_config() if algo.name != algo_name]
    _save_custom_config(algos)

def _load_builtin_config():
    path = get_builtin_config_file('builtin_algorithms.yml')
    return _load_config_file(path)

def _load_custom_config():
    path = get_config_file('registered_algorithms.yml')
    # for backward compatibility, NNI v2.5- stores all algorithms in this file
    return [algo for algo in  _load_config_file(path) if not algo.is_builtin]

def _load_config_file(path):
    with open(path, encoding='utf_8') as f:
        config = yaml.safe_load(f)
    algos = []
    for algo_type in ['tuner', 'assessor', 'advisor']:
        for algo in config.get(algo_type + 's', []):
            algos.append(AlgoMeta.load(algo, algo_type))  # type: ignore
    return algos

def _save_custom_config(custom_algos):
    config = defaultdict(list)
    for algo in custom_algos:
        config[algo.algo_type + 's'].append(algo.dump())
    text = yaml.dump(dict(config), default_flow_style=False)
    get_config_file('registered_algorithms.yml').write_text(text)
