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
    tuner_algorithm_dict = {'TPE': 'nni.hyperopt_tuner --algorithm_name tpe',\
                            'Random': 'nni.hyperopt_tuner --algorithm_name random_search',\
                            'Anneal': 'nni.hyperopt_tuner --algorithm_name anneal',\
                            'Evolution': 'nni.evolution_tuner'}


    check_empty(experiment_config, 'tuner')
    #TODO: use elegent way to detect keys
    if experiment_config['tuner'].get('tunerCommand') and experiment_config['tuner'].get('tunerCwd')\
            and (experiment_config['tuner'].get('tunerName') or experiment_config['tuner'].get('optimizationMode'))\
            or experiment_config['tuner'].get('tunerName') and experiment_config['tuner'].get('optimizationMode')\
            and (experiment_config['tuner'].get('tunerCommand') or experiment_config['tuner'].get('tunerCwd')):
        raise Exception('Please choose to use (tunerCommand, tunerCwd) or (tunerName, optimizationMode)')

    if experiment_config['tuner'].get('tunerCommand') and experiment_config['tuner'].get('tunerCwd'):
        check_directory(experiment_config['tuner'], 'tunerCwd')
        experiment_config['tuner']['tunerCwd'] = os.path.abspath(experiment_config['tuner']['tunerCwd'])
    elif experiment_config['tuner'].get('tunerName') and experiment_config['tuner'].get('optimizationMode'):
        check_choice(experiment_config['tuner'], 'tunerName', ['TPE', 'Random', 'Anneal', 'Evolution'])
        check_choice(experiment_config['tuner'], 'optimizationMode', ['Maximize', 'Minimize'])
        if experiment_config['tuner']['optimizationMode'] == 'Maximize':
            experiment_config['tuner']['optimizationMode'] = 'maximize'
        else:
            experiment_config['tuner']['optimizationMode'] = 'minimize'

        experiment_config['tuner']['tunerCommand'] = 'python3 -m %s --optimize_mode %s'\
                                                     % (tuner_algorithm_dict.get(experiment_config['tuner']['tunerName']), experiment_config['tuner']['optimizationMode'])
        experiment_config['tuner']['tunerCwd'] = ''
    else:
        raise ValueError('Please complete tuner information!')

    if experiment_config['tuner'].get('tunerGpuNum'):
        check_digit(experiment_config['tuner'], 'tunerGpuNum', 0, 100)


def validate_assessor_content(experiment_config):
    '''Validate whether assessor in experiment_config is valid'''
    assessor_algorithm_dict = {'Medianstop': 'nni.medianstop_assessor'}

    if 'assessor' in experiment_config:
        if experiment_config['assessor']:
            if experiment_config['assessor'].get('assessorCommand') and experiment_config['assessor'].get('assessorCwd')\
                    and (experiment_config['assessor'].get('assessorName') or experiment_config['assessor'].get('optimizationMode'))\
                    or experiment_config['assessor'].get('assessorName') and experiment_config['assessor'].get('optimizationMode')\
                    and (experiment_config['assessor'].get('assessorCommand') or experiment_config['assessor'].get('assessorCwd')):
                raise Exception('Please choose to use (assessorCommand, assessorCwd) or (assessorName, optimizationMode)')
            if experiment_config['assessor'].get('assessorCommand') and experiment_config['assessor'].get('assessorCwd'):
                check_empty(experiment_config['assessor'], 'assessorCommand')
                check_empty(experiment_config['assessor'], 'assessorCwd')
                check_directory(experiment_config['assessor'], 'assessorCwd')
                experiment_config['assessor']['assessorCwd'] = os.path.abspath(experiment_config['assessor']['assessorCwd'])
                if 'assessorGpuNum' in experiment_config['assessor']:
                    if experiment_config['assessor']['assessorGpuNum']:
                        check_digit(experiment_config['assessor'], 'assessorGpuNum', 0, 100)
            elif experiment_config['assessor'].get('assessorName') and experiment_config['assessor'].get('optimizationMode'):
                check_choice(experiment_config['assessor'], 'assessorName', ['Medianstop'])
                check_choice(experiment_config['assessor'], 'optimizationMode', ['Maximize', 'Minimize'])
                if experiment_config['assessor']['optimizationMode'] == 'Maximize':
                    experiment_config['assessor']['optimizationMode'] = 'maximize'
                else:
                    experiment_config['assessor']['optimizationMode'] = 'minimize'

                experiment_config['assessor']['assessorCommand'] = 'python3 -m %s --optimize_mode %s'\
                        % (assessor_algorithm_dict.get(experiment_config['assessor']['assessorName']), experiment_config['assessor']['optimizationMode'])
                experiment_config['assessor']['assessorCwd'] = ''
            else:
                raise ValueError('Please complete assessor information!')
            
            if experiment_config['assessor'].get('assessorGpuNum'):
                check_digit(experiment_config['assessor'], 'assessorGpuNum', 0, 100)


def validate_trail_content(experiment_config):
    '''Validate whether trial in experiment_config is valid'''
    check_empty(experiment_config, 'trial')
    check_empty(experiment_config['trial'], 'trialCommand')
    check_empty(experiment_config['trial'], 'trialCodeDir')
    check_directory(experiment_config['trial'], 'trialCodeDir')
    experiment_config['trial']['trialCodeDir'] = os.path.abspath(experiment_config['trial']['trialCodeDir'])
    check_empty(experiment_config['trial'], 'trialGpuNum')
    check_digit(experiment_config['trial'], 'trialGpuNum', 0, 100)


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
        check_empty(machine, 'passwd')


def validate_annotation_content(experiment_config):
    '''Valid whether useAnnotation and searchSpacePath is coexist'''
    if experiment_config.get('useAnnotation'):
        if experiment_config.get('searchSpacePath'):
            print('searchSpacePath', experiment_config.get('searchSpacePath'))
            raise Exception('If you set useAnnotation=true, please leave searchSpacePath empty')
    else:
        # validate searchSpaceFile
        check_empty(experiment_config, 'searchSpacePath')
        check_file(experiment_config, 'searchSpacePath')


def validate_all_content(experiment_config):
    '''Validate whether experiment_config is valid'''
    validate_common_content(experiment_config)
    validate_tuner_content(experiment_config)
    validate_assessor_content(experiment_config)
    validate_trail_content(experiment_config)
    # validate_annotation_content(experiment_config)
    if experiment_config['trainingServicePlatform'] == 'remote':
        validate_machinelist_content(experiment_config)
