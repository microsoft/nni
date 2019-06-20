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
from .config_schema import LOCAL_CONFIG_SCHEMA, REMOTE_CONFIG_SCHEMA, PAI_CONFIG_SCHEMA, KUBEFLOW_CONFIG_SCHEMA, FRAMEWORKCONTROLLER_CONFIG_SCHEMA, \
tuner_schema_dict, advisor_schema_dict, assessor_schema_dict
from schema import SchemaMissingKeyError, SchemaForbiddenKeyError, SchemaUnexpectedTypeError, SchemaWrongKeyError, SchemaError
from .common_utils import get_json_content, print_error, print_warning, print_normal
from schema import Schema, And, Use, Optional, Regex, Or

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
        print_error('the unit of time could only from {s, m, h, d}')
        exit(1)
    time = time[:-1]
    if not time.isdigit():
        print_error('time format error!')
        exit(1)
    parse_dict = {'s':1, 'm':60, 'h':3600, 'd':86400}
    return int(time) * parse_dict[unit]

def parse_path(experiment_config, config_path):
    '''Parse path in config file'''
    expand_path(experiment_config, 'searchSpacePath')
    if experiment_config.get('trial'):
        expand_path(experiment_config['trial'], 'codeDir')
    if experiment_config.get('tuner'):
        expand_path(experiment_config['tuner'], 'codeDir')
    if experiment_config.get('assessor'):
        expand_path(experiment_config['assessor'], 'codeDir')
    if experiment_config.get('advisor'):
        expand_path(experiment_config['advisor'], 'codeDir')

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
    if experiment_config.get('advisor'):
        parse_relative_path(root_path, experiment_config['advisor'], 'codeDir')
    if experiment_config.get('machineList'):
        for index in range(len(experiment_config['machineList'])):
            parse_relative_path(root_path, experiment_config['machineList'][index], 'sshKeyPath')

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
    if experiment_config.get('kubeflowConfig'):
        if experiment_config.get('kubeflowConfig').get('operator') == 'tf-operator':
            if experiment_config.get('trial').get('master') is not None:
                print_error('kubeflow with tf-operator can not set master')
                exit(1)
            if experiment_config.get('trial').get('worker') is None:
                print_error('kubeflow with tf-operator must set worker')
                exit(1)
        elif experiment_config.get('kubeflowConfig').get('operator') == 'pytorch-operator':
            if experiment_config.get('trial').get('ps') is not None:
                print_error('kubeflow with pytorch-operator can not set ps')
                exit(1)
            if experiment_config.get('trial').get('master') is None:
                print_error('kubeflow with pytorch-operator must set master')
                exit(1)

        if experiment_config.get('kubeflowConfig').get('storage') == 'nfs':
            if experiment_config.get('kubeflowConfig').get('nfs') is None:
                print_error('please set nfs configuration!')
                exit(1)
        elif experiment_config.get('kubeflowConfig').get('storage') == 'azureStorage':
            if experiment_config.get('kubeflowConfig').get('azureStorage') is None:
                print_error('please set azureStorage configuration!')
                exit(1)
        elif experiment_config.get('kubeflowConfig').get('storage') is None:
            if experiment_config.get('kubeflowConfig').get('azureStorage'):
                print_error('please set storage type!')
                exit(1)

def validate_common_content(experiment_config):
    '''Validate whether the common values in experiment_config is valid'''
    if not experiment_config.get('trainingServicePlatform') or \
        experiment_config.get('trainingServicePlatform') not in ['local', 'remote', 'pai', 'kubeflow', 'frameworkcontroller']:
        print_error('Please set correct trainingServicePlatform!')
        exit(1)
    schema_dict = {
            'local': LOCAL_CONFIG_SCHEMA,
            'remote': REMOTE_CONFIG_SCHEMA,
            'pai': PAI_CONFIG_SCHEMA,
            'kubeflow': KUBEFLOW_CONFIG_SCHEMA,
            'frameworkcontroller': FRAMEWORKCONTROLLER_CONFIG_SCHEMA
        }
    separate_schema_dict = {
        'tuner': tuner_schema_dict,
        'advisor': advisor_schema_dict,
        'assessor': assessor_schema_dict
    }
    separate_builtInName_dict = {
        'tuner': 'builtinTunerName',
        'advisor': 'builtinAdvisorName',
        'assessor': 'builtinAssessorName'
    }
    try:
        schema_dict.get(experiment_config['trainingServicePlatform']).validate(experiment_config)
        for separate_key in separate_schema_dict.keys():
            if experiment_config.get(separate_key):
                if experiment_config[separate_key].get(separate_builtInName_dict[separate_key]):
                    validate = False
                    for key in separate_schema_dict[separate_key].keys():
                        if key.__contains__(experiment_config[separate_key][separate_builtInName_dict[separate_key]]):
                            Schema({**separate_schema_dict[separate_key][key]}).validate(experiment_config[separate_key])
                            validate = True
                            break
                    if not validate:
                        print_error('%s %s error!' % (separate_key, separate_builtInName_dict[separate_key]))
                        exit(1)
                else:
                    Schema({**separate_schema_dict[separate_key]['customized']}).validate(experiment_config[separate_key])
    except SchemaError as error:
        print_error('Your config file is not correct, please check your config file content!')
        if error.__str__().__contains__('Wrong key'):
            print_error(' '.join(error.__str__().split()[:3]))
        else:
            print_error(error)
        exit(1)

    #set default value
    if experiment_config.get('maxExecDuration') is None:
        experiment_config['maxExecDuration'] = '999d'
    if experiment_config.get('maxTrialNum') is None:
        experiment_config['maxTrialNum'] = 99999
    if experiment_config['trainingServicePlatform'] == 'remote':
        for index in range(len(experiment_config['machineList'])):
            if experiment_config['machineList'][index].get('port') is None:
                experiment_config['machineList'][index]['port'] = 22

def validate_customized_file(experiment_config, spec_key):
    '''
    check whether the file of customized tuner/assessor/advisor exists
    spec_key: 'tuner', 'assessor', 'advisor'
    '''
    if experiment_config[spec_key].get('codeDir') and \
        experiment_config[spec_key].get('classFileName') and \
        experiment_config[spec_key].get('className'):
        if not os.path.exists(os.path.join(
                experiment_config[spec_key]['codeDir'],
                experiment_config[spec_key]['classFileName'])):
            print_error('%s file directory is not valid!'%(spec_key))
            exit(1)
    else:
        print_error('%s file directory is not valid!'%(spec_key))
        exit(1)

def parse_tuner_content(experiment_config):
    '''Validate whether tuner in experiment_config is valid'''
    if experiment_config['tuner'].get('builtinTunerName'):
        experiment_config['tuner']['className'] = experiment_config['tuner']['builtinTunerName']
    else:
        validate_customized_file(experiment_config, 'tuner')

def parse_assessor_content(experiment_config):
    '''Validate whether assessor in experiment_config is valid'''
    if experiment_config.get('assessor'):
        if experiment_config['assessor'].get('builtinAssessorName'):
            experiment_config['assessor']['className'] = experiment_config['assessor']['builtinAssessorName']
        else:
            validate_customized_file(experiment_config, 'assessor')

def parse_advisor_content(experiment_config):
    '''Validate whether advisor in experiment_config is valid'''
    if experiment_config['advisor'].get('builtinAdvisorName'):
        experiment_config['advisor']['className'] = experiment_config['advisor']['builtinAdvisorName']
    else:
        validate_customized_file(experiment_config, 'advisor')

def validate_annotation_content(experiment_config, spec_key, builtin_name):
    '''
    Valid whether useAnnotation and searchSpacePath is coexist
    spec_key: 'advisor' or 'tuner'
    builtin_name: 'builtinAdvisorName' or 'builtinTunerName'
    '''
    if experiment_config.get('useAnnotation'):
        if experiment_config.get('searchSpacePath'):
            print_error('If you set useAnnotation=true, please leave searchSpacePath empty')
            exit(1)
    else:
        # validate searchSpaceFile
        if experiment_config[spec_key].get(builtin_name) == 'NetworkMorphism':
            return
        if experiment_config[spec_key].get(builtin_name):
            if experiment_config.get('searchSpacePath') is None:
                print_error('Please set searchSpacePath!')
                exit(1)
            validate_search_space_content(experiment_config)

def validate_machine_list(experiment_config):
    '''Validate machine list'''
    if experiment_config.get('trainingServicePlatform') == 'remote' and experiment_config.get('machineList') is None:
        print_error('Please set machineList!')
        exit(1)

def validate_pai_trial_conifg(experiment_config):
    '''validate the trial config in pai platform'''
    if experiment_config.get('trainingServicePlatform') == 'pai':
        if experiment_config.get('trial').get('shmMB') and \
        experiment_config['trial']['shmMB'] > experiment_config['trial']['memoryMB']:
            print_error('shmMB should be no more than memoryMB!')
            exit(1)

def validate_all_content(experiment_config, config_path):
    '''Validate whether experiment_config is valid'''
    parse_path(experiment_config, config_path)
    validate_common_content(experiment_config)
    validate_pai_trial_conifg(experiment_config)
    experiment_config['maxExecDuration'] = parse_time(experiment_config['maxExecDuration'])
    if experiment_config.get('advisor'):
        if experiment_config.get('assessor') or experiment_config.get('tuner'):
            print_error('advisor could not be set with assessor or tuner simultaneously!')
            exit(1)
        parse_advisor_content(experiment_config)
        validate_annotation_content(experiment_config, 'advisor', 'builtinAdvisorName')
    else:
        if not experiment_config.get('tuner'):
            raise Exception('Please provide tuner spec!')
        parse_tuner_content(experiment_config)
        parse_assessor_content(experiment_config)
        validate_annotation_content(experiment_config, 'tuner', 'builtinTunerName')
