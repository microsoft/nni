# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict
import json
import pkginfo
import nni
from nni.tools.package_utils import read_installed_package_meta, get_installed_package_meta, \
    write_package_meta, get_builtin_algo_meta, get_not_installable_builtin_names, ALGO_TYPES

from .constants import INSTALLABLE_PACKAGE_META
from .common_utils import print_error, print_green
from .command_utils import install_requirements_command, call_pip_install, call_pip_uninstall

PACKAGE_TYPES = ['tuner', 'assessor', 'advisor']

def install_by_name(package_name):
    if package_name not in INSTALLABLE_PACKAGE_META:
        raise RuntimeError('{} is not found in installable packages!'.format(package_name))

    requirements_path = os.path.join(nni.__path__[0], 'algorithms/hpo', INSTALLABLE_PACKAGE_META[package_name]['code_sub_dir'], 'requirements.txt')
    assert os.path.exists(requirements_path)

    return install_requirements_command(requirements_path)

def package_install(args):
    '''install packages'''
    installed = False
    try:
        if args.name:
            if install_by_name(args.name) == 0:
                package_meta = {}
                package_meta['type'] = INSTALLABLE_PACKAGE_META[args.name]['type']
                package_meta['name'] = args.name
                package_meta['class_name'] = INSTALLABLE_PACKAGE_META[args.name]['class_name']
                package_meta['class_args_validator'] = INSTALLABLE_PACKAGE_META[args.name]['class_args_validator']
                save_package_meta_data(package_meta)
                print_green('{} installed!'.format(args.name))
                installed = True
        else:
            package_meta = get_nni_meta(args.source)
            if package_meta:
                if call_pip_install(args.source) == 0:
                    save_package_meta_data(package_meta)
                    print_green('{} installed!'.format(package_meta['name']))
                    installed = True
    except Exception as e:
        print_error(e)
    if not installed:
        print_error('installation failed!')

def package_uninstall(args):
    '''uninstall packages'''
    name = args.name[0]
    if name in get_not_installable_builtin_names():
        print_error('{} can not be uninstalled!'.format(name))
        exit(1)
    meta = get_installed_package_meta(None, name)
    if meta is None:
        print_error('package {} not found!'.format(name))
        return
    if 'installed_package' in meta:
        call_pip_uninstall(meta['installed_package'])
    if remove_package_meta_data(name):
        print_green('{} uninstalled sucessfully!'.format(name))
    else:
        print_error('Failed to uninstall {}!'.format(name))

def package_show(args):
    '''show specified packages'''
    builtin_name = args.name[0]
    meta = get_builtin_algo_meta(builtin_name=builtin_name)
    if meta:
        print(json.dumps(meta, indent=4))
    else:
        print_error('package {} not found'.format(builtin_name))

def print_package_list(meta):
    print('+-----------------+------------+-----------+--------=-------------+------------------------------------------+')
    print('|      Name       |    Type    | Installed |      Class Name      |               Module Name                |')
    print('+-----------------+------------+-----------+----------------------+------------------------------------------+')
    MAX_MODULE_NAME = 38
    for t in ['tuners', 'assessors', 'advisors']:
        for p in meta[t]:
            module_name = '.'.join(p['class_name'].split('.')[:-1])
            if len(module_name) > MAX_MODULE_NAME:
                module_name = module_name[:MAX_MODULE_NAME-3] + '...'
            class_name = p['class_name'].split('.')[-1]
            print('| {:15s} | {:10s} | {:9s} | {:20s} | {:40s} |'.format(p['name'], t, p['installed'], class_name, module_name[:38]))
    print('+-----------------+------------+-----------+----------------------+------------------------------------------+')

def package_list(args):
    '''list all packages'''
    if args.all:
        meta = get_builtin_algo_meta()
    else:
        meta = read_installed_package_meta()

    installed_names = defaultdict(list)
    for t in ['tuners', 'assessors', 'advisors']:
        for p in meta[t]:
            p['installed'] = 'Yes'
            installed_names[t].append(p['name'])
    for k, v in INSTALLABLE_PACKAGE_META.items():
        t = v['type']+'s'
        if k not in installed_names[t]:
            meta[t].append({
                'name': k,
                'class_name': v['class_name'],
                'class_args_validator': v['class_args_validator'],
                'installed': 'No'
            })

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

def remove_package_meta_data(name):
    config = read_installed_package_meta()

    updated = False
    for t in ALGO_TYPES:
        for meta in config[t]:
            if meta['name'] == name:
                config[t].remove(meta)
                updated = True
    if updated:
        write_package_meta(config)
        return True
    return False

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
