# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict
import json
import pkginfo
import nni
from nni.tools.package_utils import read_installed_package_meta, get_installed_package_meta, \
    write_package_meta, ALGO_TYPES
from .common_utils import print_error, print_green, get_yml_content

PACKAGE_TYPES = ['tuner', 'assessor', 'advisor']

def read_reg_meta_list(meta_path):
    content = get_yml_content(meta_path)
    if content.get('algorithms'):
        meta_list = content.get('algorithms')
    else:
        meta_list = [content]
    for meta in meta_list:
        assert 'algoType' in meta
        assert 'builtinName' in meta
        assert 'className' in meta
    return meta_list

def algo_reg(args):
    print(args)
    meta_list = read_reg_meta_list(args.meta_path)
    print(meta_list)
    for meta in meta_list:
        save_package_meta_data(meta)

def algo_unreg(args):
    '''uninstall packages'''
    name = args.name[0]
    meta = get_installed_package_meta(None, name)
    print(meta)
    if meta is None:
        print_error('package {} not found!'.format(name))
        return
    if remove_package_meta_data(name):
        print_green('{} uninstalled sucessfully!'.format(name))
    else:
        print_error('Failed to uninstall {}!'.format(name))

def algo_show(args):
    '''show specified packages'''
    builtin_name = args.name[0]
    meta = get_installed_package_meta(None, builtin_name)
    if meta:
        print(json.dumps(meta, indent=4))
    else:
        print_error('package {} not found'.format(builtin_name))

def algo_list(args):
    meta = read_installed_package_meta()
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


def save_package_meta_data(meta_data):
    assert meta_data['algoType'] in PACKAGE_TYPES
    assert 'builtinName' in meta_data
    assert 'className' in meta_data
    meta_data['source'] = 'user'

    config = read_installed_package_meta()

    if meta_data['builtinName'] in [x['builtinName'] for x in config[meta_data['algoType']+'s']]:
        raise ValueError('builtinName %s already installed' % meta_data['builtinName'])

    config[meta_data['algoType']+'s'].append(meta_data)
    write_package_meta(config)

def remove_package_meta_data(name):
    config = read_installed_package_meta()

    updated = False
    for t in ALGO_TYPES:
        for meta in config[t]:
            if meta['builtinName'] == name:
                config[t].remove(meta)
                updated = True
    if updated:
        write_package_meta(config)
        return True
    return False
