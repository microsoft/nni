# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import json
import nni
from nni.tools import package_utils
from .common_utils import print_error, print_green, get_yml_content

def read_reg_meta_list(meta_path):
    content = get_yml_content(meta_path)
    if content.get('algorithms'):
        meta_list = content.get('algorithms')
    else:
        meta_list = [content]
    for meta in meta_list:
        assert 'algoType' in meta
        assert meta['algoType'] in ['tuner', 'assessor', 'advisor']
        assert 'builtinName' in meta
        assert 'className' in meta
        meta['nniVersion'] = nni.__version__
    return [package_utils.AlgoMeta.load(meta) for meta in meta_list]

def verify_algo_import(meta):
    def _do_verify_import(full_name):
        module_name, class_name = full_name.rsplit('.', 1)
        class_module = importlib.import_module(module_name)
        getattr(class_module, class_name)

    _do_verify_import(meta.class_name)

    if meta.validator_class_name is not None:
        _do_verify_import(meta.validator_class_name)

def algo_reg(args):
    meta_list = read_reg_meta_list(args.meta_path)
    for meta in meta_list:
        old = package_utils.get_algo_meta(meta.name)
        if old is not None and old.is_builtin:
            print_error(f'Cannot overwrite builtin algorithm {meta.name}')
            continue

        verify_algo_import(meta)
        if old is not None:
            print_green(f'Updating exist algorithm')
        package_utils.register_algo_meta(meta)
        print_green(f'{meta.name} registered sucessfully!')

def algo_unreg(args):
    name = args.name[0]
    meta = package_utils.get_algo_meta(name)
    if meta is None:
        print_error('builtin algorithms {} not found!'.format(name))
        return
    if meta.is_builtin:
        print_error('{} is provided by nni, can not be unregistered!'.format(name))
        return
    package_utils.unregister_algo_meta(name)
    print_green('{} unregistered sucessfully!'.format(name))

def algo_show(args):
    builtin_name = args.name[0]
    meta = package_utils.get_algo_meta(builtin_name)
    if meta is not None:
        print(json.dumps(meta.dump(), indent=4))
    else:
        print_error('package {} not found'.format(builtin_name))

def algo_list(args):
    print('+-----------------+------------+-----------+--------=-------------+------------------------------------------+')
    print('|      Name       |    Type    |   source  |      Class Name      |               Module Name                |')
    print('+-----------------+------------+-----------+----------------------+------------------------------------------+')
    MAX_MODULE_NAME = 38
    for meta in package_utils.get_all_algo_meta():
        module_name, class_name = meta.class_name.rsplit('.', 1)
        if len(module_name) > MAX_MODULE_NAME:
            module_name = module_name[:MAX_MODULE_NAME-3] + '...'
        fields = [
            meta.name,
            meta.algo_type,
            'nni' if meta.is_builtin else 'user',
            class_name,
            module_name
        ]
        print('| {:15s} | {:10s} | {:9s} | {:20s} | {:40s} |'.format(*fields))
    print('+-----------------+------------+-----------+----------------------+------------------------------------------+')
