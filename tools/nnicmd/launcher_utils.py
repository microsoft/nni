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

def expand_path(experiment_config, key):
    '''Change '~' to user home directory'''
    if experiment_config.get(key):
        experiment_config[key] = os.path.expanduser(experiment_config[key])

def parse_relative_path(root_path, experiment_config, key):
    '''Change relative path to absolute path'''
    if experiment_config.get(key) and not os.path.isabs(experiment_config.get(key)):
        experiment_config[key] = os.path.join(root_path, experiment_config.get(key))

def check_empty(experiment_config, key):
    '''Check whether a key is in experiment_config and has non-empty value'''
    if key not in experiment_config or experiment_config[key] is None:
        raise ValueError('%s can not be empty' % key)

def check_digit(experiment_config, key, start, end):
    '''Check whether a value in experiment_config is digit and in a range of [start, end]'''
    if not str(experiment_config[key]).isdigit() or experiment_config[key] < start or \
            experiment_config[key] > end:
        raise ValueError('%s must be a digit from %s to %s' % (key, start, end))


def check_directory(experiment_config, key):
    '''Check whether a value in experiment_config is a valid directory'''
    if not os.path.isdir(experiment_config[key]):
        raise NotADirectoryError('%s is not a valid directory' % key)


def check_file(experiment_config, key):
    '''Check whether a value in experiment_config is a valid file'''
    if not os.path.exists(experiment_config[key]):
        raise FileNotFoundError('%s is not a valid file path' % key)


def check_choice(experiment_config, key, choice_list):
    '''Check whether a value in experiment_config is in a choice list'''
    if not experiment_config[key] in choice_list:
        raise ValueError('%s must in [%s]' % (key, ','.join(choice_list)))

def parse_time(experiment_config, key):
    '''Parse time format'''
    unit = experiment_config[key][-1]
    if unit not in ['s', 'm', 'h', 'd']:
        raise ValueError('the unit of time could only from {s, m, h, d}')
    time = experiment_config[key][:-1]
    if not time.isdigit():
        raise ValueError('time format error!')
    parse_dict = {'s':1, 'm':60, 'h':3600, 'd':86400}
    experiment_config[key] = int(time) * parse_dict[unit]

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
                raise ValueError('please use _type and _value to specify searchspace!')
    except:
        raise Exception('searchspace file is not a valid json format!')



def validate_common_content(experiment_config):
    '''Validate whether the common values in experiment_config is valid'''
    #validate authorName
    check_empty(experiment_config, 'authorName')

    #validate experimentName
    check_empty(experiment_config, 'experimentName')

    #validate trialNoncurrency
    check_empty(experiment_config, 'trialConcurrency')
    check_digit(experiment_config, 'trialConcurrency', 1, 1000)

    #validate execDuration
    check_empty(experiment_config, 'maxExecDuration')
    parse_time(experiment_config, 'maxExecDuration')

    #validate maxTrialNum
    check_empty(experiment_config, 'maxTrialNum')
    check_digit(experiment_config, 'maxTrialNum', 1, 1000)

    #validate trainingService
    check_empty(experiment_config, 'trainingServicePlatform')
    check_choice(experiment_config, 'trainingServicePlatform', ['local', 'remote'])


def validate_tuner_content(experiment_config):
    '''Validate whether tuner in experiment_config is valid'''
    tuner_class_name_dict = {'TPE': 'HyperoptTuner',\
                            'Random': 'HyperoptTuner',\
                            'Anneal': 'HyperoptTuner',\
                            'Evolution': 'EvolutionTuner'}

    tuner_algorithm_name_dict = {'TPE': 'tpe',\
                            'Random': 'random_search',\
                            'Anneal': 'anneal'}

    if experiment_config.get('tuner') is None:
        raise ValueError('Please set tuner!')
    if (experiment_config['tuner'].get('builtinTunerName') and \
        (experiment_config['tuner'].get('codeDir') or experiment_config['tuner'].get('classFileName') or experiment_config['tuner'].get('className'))) or \
        (experiment_config['tuner'].get('codeDir') and experiment_config['tuner'].get('classFileName') and experiment_config['tuner'].get('className') and \
        experiment_config['tuner'].get('builtinTunerName')):
            raise ValueError('Please check tuner content!')
    
    if experiment_config['tuner'].get('builtinTunerName') and experiment_config['tuner'].get('classArgs'):
        if tuner_class_name_dict.get(experiment_config['tuner']['builtinTunerName']) is None:
            raise ValueError('Please set correct builtinTunerName!')
        experiment_config['tuner']['className'] = tuner_class_name_dict.get(experiment_config['tuner']['builtinTunerName'])
        if experiment_config['tuner']['classArgs'].get('optimize_mode') is None:
            raise ValueError('Please set optimize_mode!')
        if experiment_config['tuner']['classArgs']['optimize_mode'] not in ['maximize', 'minimize']:
            raise ValueError('optimize_mode should be maximize or minimize')
        if tuner_algorithm_name_dict.get(experiment_config['tuner']['builtinTunerName']) and \
            tuner_algorithm_name_dict.get(experiment_config['tuner']['builtinTunerName']):
            experiment_config['tuner']['classArgs']['algorithm_name'] = tuner_algorithm_name_dict.get(experiment_config['tuner']['builtinTunerName'])
    elif experiment_config['tuner'].get('codeDir') and experiment_config['tuner'].get('classFileName') and experiment_config['tuner'].get('className'):
        if not os.path.exists(os.path.join(experiment_config['tuner']['codeDir'], experiment_config['tuner']['classFileName'])):
            raise ValueError('Tuner file directory is not valid!')
    else:
        raise ValueError('Tuner format is not valid!')

    if experiment_config['tuner'].get('gpuNum'):
        check_digit(experiment_config['tuner'], 'gpuNum', 0, 100)


def validate_assessor_content(experiment_config):
    '''Validate whether assessor in experiment_config is valid'''
    assessor_class_name_dict = {'Medianstop': 'MedianstopAssessor'}
        
    if experiment_config.get('assessor'):
        if (experiment_config['assessor'].get('builtinAssessorName') and \
            (experiment_config['assessor'].get('codeDir') or experiment_config['assessor'].get('classFileName') or experiment_config['assessor'].get('className'))) or \
            (experiment_config['assessor'].get('codeDir') and experiment_config['assessor'].get('classFileName') and experiment_config['assessor'].get('className') and \
            experiment_config['assessor'].get('builtinAssessorName')):
                raise ValueError('Please check assessor content!')
        
        if experiment_config['assessor'].get('builtinAssessorName') and experiment_config['assessor'].get('classArgs'):
            if assessor_class_name_dict.get(experiment_config['assessor']['builtinAssessorName']) is None:
                raise ValueError('Please set correct builtinAssessorName!')
            experiment_config['assessor']['className'] = assessor_class_name_dict.get(experiment_config['assessor']['builtinAssessorName'])
            if experiment_config['assessor']['classArgs'].get('optimize_mode') is None:
                raise ValueError('Please set optimize_mode!')
            if experiment_config['assessor']['classArgs']['optimize_mode'] not in ['maximize', 'minimize']:
                raise ValueError('optimize_mode should be maximize or minimize')
        elif experiment_config['assessor'].get('codeDir') and experiment_config['assessor'].get('classFileName') and experiment_config['assessor'].get('className'):
            if not os.path.exists(os.path.join(experiment_config['assessor']['codeDir'], experiment_config['assessor']['classFileName'])):
                raise ValueError('Assessor file directory is not valid!')
        else:
            raise ValueError('Assessor format is not valid!')
                
        if experiment_config['assessor'].get('gpuNum'):
            check_digit(experiment_config['assessor'], 'gpuNum', 0, 100)


def validate_trail_content(experiment_config):
    '''Validate whether trial in experiment_config is valid'''
    check_empty(experiment_config, 'trial')
    check_empty(experiment_config['trial'], 'command')
    check_empty(experiment_config['trial'], 'codeDir')
    check_directory(experiment_config['trial'], 'codeDir')
    experiment_config['trial']['codeDir'] = os.path.abspath(experiment_config['trial']['codeDir'])
    if experiment_config['trial'].get('gpuNum') is None:
        experiment_config['trial']['gpuNum'] = 0
    else:
        check_digit(experiment_config['trial'], 'gpuNum', 0, 100)


def validate_machinelist_content(experiment_config):
    '''Validate whether meachineList in experiment_config is valid'''
    check_empty(experiment_config, 'machineList')
    for i, machine in enumerate(experiment_config['machineList']):
        check_empty(machine, 'ip')
        if machine.get('port') is None:
            experiment_config['machineList'][i]['port'] = 22
        else:
            check_digit(machine, 'port', 0, 65535)
        check_empty(machine, 'username')
        if machine.get('passwd') is None and machine.get('sshKeyPath') is None:
            raise ValueError('Please set passwd or sshKeyPath for remote machine!')
        if machine.get('sshKeyPath') is None and machine.get('passphrase'):
            raise ValueError('Please set sshKeyPath!')
        if machine.get('sshKeyPath'):
            check_file(machine, 'sshKeyPath')


def validate_annotation_content(experiment_config):
    '''Valid whether useAnnotation and searchSpacePath is coexist'''
    if experiment_config.get('useAnnotation'):
        if experiment_config.get('searchSpacePath'):
            print('searchSpacePath', experiment_config.get('searchSpacePath'))
            raise Exception('If you set useAnnotation=true, please leave searchSpacePath empty')
    else:
        # validate searchSpaceFile
        if experiment_config['tuner'].get('tunerName') and experiment_config['tuner'].get('optimizationMode'):
            check_empty(experiment_config, 'searchSpacePath')
            check_file(experiment_config, 'searchSpacePath')
            validate_search_space_content(experiment_config)


def validate_all_content(experiment_config, config_path):
    '''Validate whether experiment_config is valid'''
    parse_path(experiment_config, config_path)
    validate_common_content(experiment_config)
    validate_tuner_content(experiment_config)
    validate_assessor_content(experiment_config)
    validate_trail_content(experiment_config)
    validate_annotation_content(experiment_config)
    if experiment_config['trainingServicePlatform'] == 'remote':
        validate_machinelist_content(experiment_config)
