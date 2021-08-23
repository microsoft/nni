# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import importlib
import json
from nni.tools.package_utils import read_registerd_algo_meta, get_registered_algo_meta, \
    write_registered_algo_meta, ALGO_TYPES, parse_full_class_name
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
    return meta_list

def verify_algo_import(meta):
    def _do_verify_import(fullName):
        module_name, class_name = parse_full_class_name(fullName)
        class_module = importlib.import_module(module_name)
        getattr(class_module, class_name)

    _do_verify_import(meta['className'])

    if meta.get('classArgsValidator'):
        _do_verify_import(meta['classArgsValidator'])

def algo_reg(args):
    meta_list = read_reg_meta_list(args.meta_path)
    for meta in meta_list:
        if get_registered_algo_meta(meta['builtinName']) is not None:
            print_error('builtinName {} already registered'.format(meta['builtinName']))
            return
        verify_algo_import(meta)
        save_algo_meta_data(meta)
        print_green('{} registered sucessfully!'.format(meta['builtinName']))

def algo_unreg(args):
    name = args.name[0]
    meta = get_registered_algo_meta(name)
    if meta is None:
        print_error('builtin algorithms {} not found!'.format(name))
        return
    if meta['source'] == 'nni':
        print_error('{} is provided by nni, can not be unregistered!'.format(name))
        return
    if remove_algo_meta_data(name):
        print_green('{} unregistered sucessfully!'.format(name))
    else:
        print_error('Failed to unregistered {}!'.format(name))

def algo_show(args):
    builtin_name = args.name[0]
    meta = get_registered_algo_meta(builtin_name)
    if meta:
        print(json.dumps(meta, indent=4))
    else:
        print_error('package {} not found'.format(builtin_name))

def algo_list(args):
    meta = read_registerd_algo_meta()
    print('+-----------------+------------+-----------+--------=-------------+------------------------------------------+')
    print('|      Name       |    Type    |   source  |      Class Name      |               Module Name                |')
    print('+-----------------+------------+-----------+----------------------+------------------------------------------+')
    MAX_MODULE_NAME = 38
    for t in ['tuners', 'assessors', 'advisors']:
        for p in meta[t]:
            module_name = '.'.join(p['className'].split('.')[:-1])
            if len(module_name) > MAX_MODULE_NAME:
                module_name = module_name[:MAX_MODULE_NAME-3] + '...'
            class_name = p['className'].split('.')[-1]
            print('| {:15s} | {:10s} | {:9s} | {:20s} | {:40s} |'.format(p['builtinName'], t, p['source'], class_name, module_name[:38]))
    print('+-----------------+------------+-----------+----------------------+------------------------------------------+')


def save_algo_meta_data(meta_data):
    meta_data['source'] = 'user'
    config = read_registerd_algo_meta()
    config[meta_data['algoType']+'s'].append(meta_data)
    write_registered_algo_meta(config)

def remove_algo_meta_data(name):
    config = read_registerd_algo_meta()

    updated = False
    for t in ALGO_TYPES:
        for meta in config[t]:
            if meta['builtinName'] == name:
                config[t].remove(meta)
                updated = True
    if updated:
        write_registered_algo_meta(config)
        return True
    return False
