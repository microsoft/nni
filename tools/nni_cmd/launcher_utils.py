# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from schema import SchemaError
from .config_schema import NNIConfigSchema
from .common_utils import print_normal

def expand_path(experiment_config, key):
    '''Change '~' to user home directory'''
    if experiment_config.get(key):
        experiment_config[key] = os.path.expanduser(experiment_config[key])

def parse_relative_path(root_path, experiment_config, key):
    '''Change relative path to absolute path'''
    if experiment_config.get(key) and not os.path.isabs(experiment_config.get(key)):
        absolute_path = os.path.join(root_path, experiment_config.get(key))
        print_normal('expand %s: %s to %s ' % (key, experiment_config[key], absolute_path))
        experiment_config[key] = absolute_path

def parse_time(time):
    '''Change the time to seconds'''
    unit = time[-1]
    if unit not in ['s', 'm', 'h', 'd']:
        raise SchemaError('the unit of time could only from {s, m, h, d}')
    time = time[:-1]
    if not time.isdigit():
        raise SchemaError('time format error!')
    parse_dict = {'s':1, 'm':60, 'h':3600, 'd':86400}
    return int(time) * parse_dict[unit]

def parse_path(experiment_config, config_path):
    '''Parse path in config file'''
    expand_path(experiment_config, 'searchSpacePath')
    if experiment_config.get('trial'):
        expand_path(experiment_config['trial'], 'codeDir')
        if experiment_config['trial'].get('authFile'):
            expand_path(experiment_config['trial'], 'authFile')
        if experiment_config['trial'].get('ps'):
            if experiment_config['trial']['ps'].get('privateRegistryAuthPath'):
                expand_path(experiment_config['trial']['ps'], 'privateRegistryAuthPath')
        if experiment_config['trial'].get('master'):
            if experiment_config['trial']['master'].get('privateRegistryAuthPath'):
                expand_path(experiment_config['trial']['master'], 'privateRegistryAuthPath')
        if experiment_config['trial'].get('worker'):
            if experiment_config['trial']['worker'].get('privateRegistryAuthPath'):
                expand_path(experiment_config['trial']['worker'], 'privateRegistryAuthPath')
        if experiment_config['trial'].get('taskRoles'):
            for index in range(len(experiment_config['trial']['taskRoles'])):
                if experiment_config['trial']['taskRoles'][index].get('privateRegistryAuthPath'):
                    expand_path(experiment_config['trial']['taskRoles'][index], 'privateRegistryAuthPath')
    if experiment_config.get('tuner'):
        expand_path(experiment_config['tuner'], 'codeDir')
    if experiment_config.get('assessor'):
        expand_path(experiment_config['assessor'], 'codeDir')
    if experiment_config.get('advisor'):
        expand_path(experiment_config['advisor'], 'codeDir')
    if experiment_config.get('machineList'):
        for index in range(len(experiment_config['machineList'])):
            expand_path(experiment_config['machineList'][index], 'sshKeyPath')
    if experiment_config['trial'].get('paiConfigPath'):
        expand_path(experiment_config['trial'], 'paiConfigPath')

    #if users use relative path, convert it to absolute path
    root_path = os.path.dirname(config_path)
    if experiment_config.get('searchSpacePath'):
        parse_relative_path(root_path, experiment_config, 'searchSpacePath')
    if experiment_config.get('trial'):
        parse_relative_path(root_path, experiment_config['trial'], 'codeDir')
        if experiment_config['trial'].get('authFile'):
            parse_relative_path(root_path, experiment_config['trial'], 'authFile')
        if experiment_config['trial'].get('ps'):
            if experiment_config['trial']['ps'].get('privateRegistryAuthPath'):
                parse_relative_path(root_path, experiment_config['trial']['ps'], 'privateRegistryAuthPath')
        if experiment_config['trial'].get('master'):
            if experiment_config['trial']['master'].get('privateRegistryAuthPath'):
                parse_relative_path(root_path, experiment_config['trial']['master'], 'privateRegistryAuthPath')
        if experiment_config['trial'].get('worker'):
            if experiment_config['trial']['worker'].get('privateRegistryAuthPath'):
                parse_relative_path(root_path, experiment_config['trial']['worker'], 'privateRegistryAuthPath')
        if experiment_config['trial'].get('taskRoles'):
            for index in range(len(experiment_config['trial']['taskRoles'])):
                if experiment_config['trial']['taskRoles'][index].get('privateRegistryAuthPath'):
                    parse_relative_path(root_path, experiment_config['trial']['taskRoles'][index], 'privateRegistryAuthPath')
    if experiment_config.get('tuner'):
        parse_relative_path(root_path, experiment_config['tuner'], 'codeDir')
    if experiment_config.get('assessor'):
        parse_relative_path(root_path, experiment_config['assessor'], 'codeDir')
    if experiment_config.get('advisor'):
        parse_relative_path(root_path, experiment_config['advisor'], 'codeDir')
    if experiment_config.get('machineList'):
        for index in range(len(experiment_config['machineList'])):
            parse_relative_path(root_path, experiment_config['machineList'][index], 'sshKeyPath')
    if experiment_config['trial'].get('paiConfigPath'):
        parse_relative_path(root_path, experiment_config['trial'], 'paiConfigPath')

def set_default_values(experiment_config):
    if experiment_config.get('maxExecDuration') is None:
        experiment_config['maxExecDuration'] = '999d'
    if experiment_config.get('maxTrialNum') is None:
        experiment_config['maxTrialNum'] = 99999
    if experiment_config['trainingServicePlatform'] == 'remote':
        for index in range(len(experiment_config['machineList'])):
            if experiment_config['machineList'][index].get('port') is None:
                experiment_config['machineList'][index]['port'] = 22

def validate_all_content(experiment_config, config_path):
    '''Validate whether experiment_config is valid'''
    parse_path(experiment_config, config_path)
    set_default_values(experiment_config)

    NNIConfigSchema().validate(experiment_config)

    experiment_config['maxExecDuration'] = parse_time(experiment_config['maxExecDuration'])
