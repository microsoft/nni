# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict
import zipfile
import nni
import ruamel.yaml as yaml

from .constants import PACKAGE_REQUIREMENTS
from .common_utils import print_error, get_nni_config_dir
from .command_utils import install_requirements_command, call_pip_install, call_pip_uninstall

NNI_META_FILE = 'nni_config.yml'

def process_install(package_name):
    if PACKAGE_REQUIREMENTS.get(package_name) is None:
        print_error('{0} is not supported!' % package_name)
    else:
        requirements_path = os.path.join(nni.__path__[0], PACKAGE_REQUIREMENTS[package_name])
        install_requirements_command(requirements_path)

def package_install(args):
    '''install packages'''
    print(args)
    #process_install(args.name)
    if not validate_install_source(args.source):
        return
    call_pip_install(args.source)

def package_uninstall(args):
    '''install packages'''
    print(args)
    #process_install(args.name)
    call_pip_uninstall(args.source)

def package_show(args):
    '''show all packages'''
    #print(' '.join(PACKAGE_REQUIREMENTS.keys()))
    print(args)

def package_list(args):
    '''show all packages'''
    print(' '.join(PACKAGE_REQUIREMENTS.keys()))
    print(get_nni_config_dir())

    meta = {
        'type': 'tuner',
        'name': 'mytuner',
        'class_name': 'mypackage.mytuner'
    }
    #save_package_meta_data(meta)
    remove_package_meta_data('mytuner', 'tuner')

def get_package_config_path():
    config_dir = get_nni_config_dir()
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, 'installed_packages.yml')

def read_package_config():
    config_file = get_package_config_path()
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        config = defaultdict(list)
    return config

def write_package_config(config):
    config_file = get_package_config_path()
    with open(config_file, 'w') as f:
        f.write(yaml.dump(dict(config), default_flow_style=False))

def save_package_meta_data(meta_data):
    assert meta_data['type'] in ['tuner', 'assessor', 'advisor']
    assert 'name' in meta_data
    assert 'class_name' in meta_data

    config = read_package_config()

    if meta_data['name'] in [x['name'] for x in config[meta_data['type']+'s']]:
        raise ValueError('name %s already installed' % meta_data['name'])

    config[meta_data['type']+'s'].append(meta_data)
    write_package_config(config)

def remove_package_meta_data(name, class_type):
    assert class_type in ['tuner', 'assessor', 'advisor']
    config = read_package_config()

    updated = False
    for meta in config[class_type +'s']:
        if meta['name'] == name:
            config[class_type +'s'].remove(meta)
            updated = True
    if updated:
        write_package_config(config)

def validate_install_source(source):
    """
    The source should be a valid pip installation source and it contains nni_config.yml.
    """
    def check_meta_data(meta):
        print(meta)
        return True

    if not os.path.exists(source):
        print_error('{} does not exist'.format(source))
        return False
    if os.path.isdir(source):
        if not os.path.exists(os.path.join(source, 'setup.py')):
            print_error('setup.py not found')
            return False
        if not os.path.exists(os.path.join(source, NNI_META_FILE)):
            print_error('{} not found'.format(NNI_META_FILE))
            return False
        with open(os.path.join(source, NNI_META_FILE)) as f:
            meta = yaml.load(f, Loader=yaml.Loader)
            return check_meta_data(meta)
    else:
        if not source.endswith('.whl'):
            print_error('File name {} must ends with \'.whl\'')
            return False
        with zipfile.ZipFile(source) as whl_file:
            print(whl_file.namelist())
            with whl_file.open(NNI_META_FILE) as meta_file:
                meta = yaml.load(meta_file, Loader=yaml.Loader)
                return check_meta_data(meta)

    return False
