# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import pkginfo
import nni
from nni.package_utils import read_installed_package_meta, get_installed_package_meta, write_package_meta, get_builtin_algo_meta

from .constants import PACKAGE_META
from .common_utils import print_error
from .command_utils import install_requirements_command, call_pip_install, call_pip_uninstall

PACKAGE_TYPES = ['tuner', 'assessor', 'advisor']

def install_by_name(package_name):
    if package_name not in PACKAGE_META:
        print_error('{} is not supported!'.format(package_name))
        return -1

    requirements_path = os.path.join(nni.__path__[0], PACKAGE_META[package_name]['code_sub_dir'], 'requirements.txt')
    assert os.path.exists(requirements_path)

    return install_requirements_command(requirements_path)

def package_install(args):
    '''install packages'''
    if args.name:
        ret_code = install_by_name(args.name)
        if ret_code == 0:
            package_meta = {}
            package_meta['type'] = PACKAGE_META[args.name]['type']
            package_meta['name'] = args.name
            package_meta['class_name'] = PACKAGE_META[args.name]['class_name']
            package_meta['class_args_validator'] = PACKAGE_META[args.name]['class_args_validator']
            save_package_meta_data(package_meta)
    else:
        package_meta = get_nni_meta(args.source)
        if package_meta:
            if call_pip_install(args.source) == 0:
                save_package_meta_data(package_meta)

def package_uninstall(args):
    '''uninstall packages'''
    meta = get_installed_package_meta(None, args.name)
    if meta is None:
        print_error('package {} not found'.format(args.name))
        return
    if 'installed_package' in meta:
        call_pip_uninstall(meta['installed_package'])
    remove_package_meta_data(args.name, args.type)

def package_show(args):
    '''show specified packages'''
    builtin_name = args.name[0]
    meta = get_builtin_algo_meta(builtin_name=builtin_name)
    if meta:
        print(json.dumps(meta, indent=4))
    else:
        print_error('package {} not found'.format(builtin_name))

def print_package_list(meta):
    print('+-----------------+------------+----------------------+------------------------------------------+')
    print('|      Name       |    Type    |      Class Name      |               Module Name                |')
    print('+-----------------+------------+----------------------+------------------------------------------+')
    MAX_MODULE_NAME = 38
    for t in ['tuners', 'assessors', 'advisors']:
        for p in meta[t]:
            module_name = '.'.join(p['class_name'].split('.')[:-1])
            if len(module_name) > MAX_MODULE_NAME:
                module_name = module_name[:MAX_MODULE_NAME-3] + '...'
            class_name = p['class_name'].split('.')[-1]
            print('| {:15s} | {:10s} | {:20s} | {:40s} |'.format(p['name'], t, class_name, module_name[:38]))
    print('+-----------------+------------+----------------------+------------------------------------------+')

def package_list(args):
    '''list all packages'''
    if args.all:
        meta = get_builtin_algo_meta()
    else:
        meta = read_installed_package_meta()
    print_package_list(meta)

def save_package_meta_data(meta_data):
    assert meta_data['type'] in PACKAGE_TYPES
    assert 'name' in meta_data
    assert 'class_name' in meta_data

    config = read_installed_package_meta()

    if meta_data['name'] in [x['name'] for x in config[meta_data['type']+'s']]:
        raise ValueError('name %s already installed' % meta_data['name'])

    package_meta = {k: meta_data[k] for k in ['name', 'class_name', 'class_args_validator'] if k in meta_data}
    if 'package_name' in meta_data:
        package_meta['installed_package'] = meta_data['package_name']
    config[meta_data['type']+'s'].append(package_meta)
    write_package_meta(config)

def remove_package_meta_data(name, class_type):
    assert class_type in PACKAGE_TYPES
    config = read_installed_package_meta()

    updated = False
    for meta in config[class_type +'s']:
        if meta['name'] == name:
            config[class_type +'s'].remove(meta)
            updated = True
    if updated:
        write_package_meta(config)

def get_nni_meta(source):
    if not os.path.exists(source):
        print_error('{} does not exist'.format(source))
        return None

    if os.path.isdir(source):
        if not os.path.exists(os.path.join(source, 'setup.py')):
            print_error('setup.py not found')
            return None
        pkg = pkginfo.Develop(source)
    else:
        if not source.endswith('.whl'):
            print_error('File name {} must ends with \'.whl\''.format(source))
            return False
        pkg = pkginfo.Wheel(source)

    classifiers = pkg.classifiers
    meta = parse_classifiers(classifiers)
    meta['package_name'] = pkg.name
    return meta

def parse_classifiers(classifiers):
    parts = []
    for c in classifiers:
        if c.startswith('NNI Package'):
            parts = [x.strip() for x in c.split('::')]
            break
    if len(parts) < 4 or not all(parts):
        raise ValueError('Can not find correct NNI meta data in package classifiers.')
    meta = {
        'type': parts[1],
        'name': parts[2],
        'class_name': parts[3]
    }
    if len(parts) >= 5:
        meta['class_args_validator'] = parts[4]

    return meta
