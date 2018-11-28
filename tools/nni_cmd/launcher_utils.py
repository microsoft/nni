# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
from .config_schema import LOCAL_CONFIG_SCHEMA, REMOTE_CONFIG_SCHEMA, PAI_CONFIG_SCHEMA, KUBEFLOW_CONFIG_SCHEMA
from .common_utils import get_json_content, print_error

def expand_path(experiment_config, key):
    '''Change '~' to user home directory'''
    if experiment_config.get(key):
        experiment_config[key] = os.path.expanduser(experiment_config[key])

def parse_relative_path(root_path, experiment_config, key):
    '''Change relative path to absolute path'''
    if experiment_config.get(key) and not os.path.isabs(experiment_config.get(key)):
        experiment_config[key] = os.path.join(root_path, experiment_config.get(key))

def parse_time(experiment_config):
    '''Parse time format'''
    unit = experiment_config['maxExecDuration'][-1]
    if unit not in ['s', 'm', 'h', 'd']:
        print_error('the unit of time could only from {s, m, h, d}')
        exit(1)
    time = experiment_config['maxExecDuration'][:-1]
    if not time.isdigit():
        print_error('time format error!')
        exit(1)
    parse_dict = {'s':1, 'm':60, 'h':3600, 'd':86400}
    experiment_config['maxExecDuration'] = int(time) * parse_dict[unit]

def parse_path(experiment_config, config_path):
    '''Parse path in config file'''
    expand_path(experiment_config, 'searchSpacePath')
    if experiment_config.get('trial'):
        expand_path(experiment_config['trial'], 'codeDir')
    if experiment_config.get('tuner'):
        expand_path(experiment_config['tuner'], 'codeDir')
    if experiment_config.get('assessor'):
        expand_path(experiment_config['assessor'], 'codeDir')
    
    #if users use relative path, convert it to absolute path
    root_path = os.path.dirname(config_path)
    if experiment_config.get('searchSpacePath'):
        parse_relative_path(root_path, experiment_config, 'searchSpacePath')
    if experiment_config.get('trial'):
        parse_relative_path(root_path, experiment_config['trial'], 'codeDir')
    if experiment_config.get('tuner'):
        parse_relative_path(root_path, experiment_config['tuner'], 'codeDir')
    if experiment_config.get('assessor'):
        parse_relative_path(root_path, experiment_config['assessor'], 'codeDir')

def validate_search_space_content(experiment_config):
    '''Validate searchspace content, 
       if the searchspace file is not json format or its values does not contain _type and _value which must be specified, 
       it will not be a valid searchspace file'''
    try:
        search_space_content = json.load(open(experiment_config.get('searchSpacePath'), 'r'))
        for value in search_space_content.values():
            if not value.get('_type') or not value.get('_value'):
                print_error('please use _type and _value to specify searchspace!')
                exit(1)
    except:
        print_error('searchspace file is not a valid json format!')
        exit(1)

def validate_kubeflow_operators(experiment_config):
    '''Validate whether the kubeflow operators are valid'''
    if experiment_config.get('kubeflowConfig').get('operator') == 'tf-operator':
        if experiment_config.get('trial').get('master') is not None:
            print_error('kubeflow with tf-operator can not set master')
            exit(1)
    elif experiment_config.get('kubeflowConfig').get('operator') == 'pytorch-operator':
        if experiment_config.get('trial').get('ps') is not None:
            print_error('kubeflow with pytorch-operator can not set ps')
            exit(1)

def validate_common_content(experiment_config):
    '''Validate whether the common values in experiment_config is valid'''
    if not experiment_config.get('trainingServicePlatform') or \
        experiment_config.get('trainingServicePlatform') not in ['local', 'remote', 'pai', 'kubeflow']:
        print_error('Please set correct trainingServicePlatform!')
        exit(0)
    schema_dict = {
            'local': LOCAL_CONFIG_SCHEMA,
            'remote': REMOTE_CONFIG_SCHEMA,
            'pai': PAI_CONFIG_SCHEMA,
            'kubeflow': KUBEFLOW_CONFIG_SCHEMA
        }
    try:
        schema_dict.get(experiment_config['trainingServicePlatform']).validate(experiment_config)
        #set default value
        if experiment_config.get('maxExecDuration') is None:
            experiment_config['maxExecDuration'] = '999d'
        if experiment_config.get('maxTrialNum') is None:
            experiment_config['maxTrialNum'] = 99999
        if experiment_config['trainingServicePlatform'] == 'remote':
            for index in range(len(experiment_config['machineList'])):
                if experiment_config['machineList'][index].get('port') is None:
                    experiment_config['machineList'][index]['port'] = 22
                
    except Exception as exception:
        print_error('Your config file is not correct, please check your config file content!\n%s' % exception)
        exit(1)

def parse_tuner_content(experiment_config):
    '''Validate whether tuner in experiment_config is valid'''
    if experiment_config['tuner'].get('builtinTunerName'):
        experiment_config['tuner']['className'] = experiment_config['tuner']['builtinTunerName']
    elif experiment_config['tuner'].get('codeDir') and \
        experiment_config['tuner'].get('classFileName') and \
        experiment_config['tuner'].get('className'):
        if not os.path.exists(os.path.join(
                experiment_config['tuner']['codeDir'],
                experiment_config['tuner']['classFileName'])):
            print_error('Tuner file directory is not valid!')
            exit(1)
    else:
        print_error('Tuner format is not valid!')
        exit(1)

def parse_assessor_content(experiment_config):
    '''Validate whether assessor in experiment_config is valid'''
    if experiment_config.get('assessor'):
        if experiment_config['assessor'].get('builtinAssessorName'):
            experiment_config['assessor']['className'] = experiment_config['assessor']['builtinAssessorName']
        elif experiment_config['assessor'].get('codeDir') and \
            experiment_config['assessor'].get('classFileName') and \
            experiment_config['assessor'].get('className'):
            if not os.path.exists(os.path.join(
                    experiment_config['assessor']['codeDir'],
                    experiment_config['assessor']['classFileName'])):
                print_error('Assessor file directory is not valid!')
                exit(1)
        else:
            print_error('Assessor format is not valid!')
            exit(1)

def validate_annotation_content(experiment_config):
    '''Valid whether useAnnotation and searchSpacePath is coexist'''
    if experiment_config.get('useAnnotation'):
        if experiment_config.get('searchSpacePath'):
            print_error('If you set useAnnotation=true, please leave searchSpacePath empty')
            exit(1)
    else:
        # validate searchSpaceFile
        if experiment_config['tuner'].get('tunerName') and experiment_config['tuner'].get('optimizationMode'):
            if experiment_config.get('searchSpacePath') is None:
                print_error('Please set searchSpace!')
                exit(1)
            validate_search_space_content(experiment_config)

def validate_machine_list(experiment_config):
    '''Validate machine list'''
    if experiment_config.get('trainingServicePlatform') == 'remote' and experiment_config.get('machineList') is None:
        print_error('Please set machineList!')
        exit(1)

def validate_all_content(experiment_config, config_path):
    '''Validate whether experiment_config is valid'''
    parse_path(experiment_config, config_path)
    validate_common_content(experiment_config)
    parse_time(experiment_config)
    parse_tuner_content(experiment_config)
    parse_assessor_content(experiment_config)
    validate_annotation_content(experiment_config)
    validate_kubeflow_operators(experiment_config)
