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
from schema import Schema, And, Use, Optional, Regex, Or
from .constants import SCHEMA_TYPE_ERROR, SCHEMA_RANGE_ERROR, SCHEMA_PATH_ERROR


def setType(key, type):
    '''check key type'''
    return And(type, error=SCHEMA_TYPE_ERROR % (key, type.__name__))

def setChoice(key, *args):
    '''check choice'''
    return And(lambda n: n in args, error=SCHEMA_RANGE_ERROR % (key, str(args)))

def setNumberRange(key, keyType, start, end):
    '''check number range'''
    return And(
        And(keyType, error=SCHEMA_TYPE_ERROR % (key, keyType.__name__)),
        And(lambda n: start <= n <= end, error=SCHEMA_RANGE_ERROR % (key, '(%s,%s)' % (start, end))),
    )

def setPathCheck(key):
    '''check if path exist'''
    return And(os.path.exists, error=SCHEMA_PATH_ERROR % key)

common_schema = {
    'authorName': setType('authorName', str),
    'experimentName': setType('experimentName', str),
    Optional('description'): setType('description', str),
    'trialConcurrency': setNumberRange('trialConcurrency', int, 1, 99999),
    Optional('maxExecDuration'): And(Regex(r'^[1-9][0-9]*[s|m|h|d]$',error='ERROR: maxExecDuration format is [digit]{s,m,h,d}')),
    Optional('maxTrialNum'): setNumberRange('maxTrialNum', int, 1, 99999),
    'trainingServicePlatform': setChoice('trainingServicePlatform', 'remote', 'local', 'pai', 'kubeflow', 'frameworkcontroller'),
    Optional('searchSpacePath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'searchSpacePath'),
    Optional('multiPhase'): setType('multiPhase', bool),
    Optional('multiThread'): setType('multiThread', bool),
    Optional('nniManagerIp'): setType('nniManagerIp', str),
    Optional('logDir'): And(os.path.isdir, error=SCHEMA_PATH_ERROR % 'logDir'),
    Optional('debug'): setType('debug', bool),
    Optional('logLevel'): setChoice('logLevel', 'trace', 'debug', 'info', 'warning', 'error', 'fatal'),
    Optional('logCollection'): setChoice('logCollection', 'http', 'none'),
    'useAnnotation': setType('useAnnotation', bool),
    Optional('tuner'): dict,
    Optional('advisor'): dict,
    Optional('assessor'): dict,
    Optional('localConfig'): {
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!')
    }
}
tuner_schema_dict = {
    ('TPE', 'Anneal', 'SMAC', 'Evolution'): {
        'builtinTunerName': setChoice('builtinTunerName', 'TPE', 'Anneal', 'SMAC', 'Evolution'),
        Optional('classArgs'): {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
        },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    ('BatchTuner', 'GridSearch', 'Random'): {
        'builtinTunerName': setChoice('builtinTunerName', 'BatchTuner', 'GridSearch', 'Random'),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    'NetworkMorphism': {
        'builtinTunerName': 'NetworkMorphism',
        'classArgs': {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('task'): setChoice('task', 'cv','nlp','common'),
            Optional('input_width'): setType('input_width', int),
            Optional('input_channel'): setType('input_channel', int),
            Optional('n_output_node'): setType('n_output_node', int),
            },
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    'MetisTuner': {
        'builtinTunerName': 'MetisTuner',
        'classArgs': {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('no_resampling'): setType('no_resampling', bool),
            Optional('no_candidates'): setType('no_candidates', bool),
            Optional('selection_num_starting_points'):  setType('selection_num_starting_points', int),
            Optional('cold_start_num'): setType('cold_start_num', int),
            },
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    'customized': {
        'codeDir': setPathCheck('codeDir'),
        'classFileName': setType('classFileName', str),
        'className': setType('className', str),
        Optional('classArgs'): dict,
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    }
}

advisor_schema_dict = {
    'Hyperband':{
        'builtinAdvisorName': Or('Hyperband'),
        'classArgs': {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('R'): setType('R', int),
            Optional('eta'): setType('eta', int)
        },
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    'BOHB':{
        'builtinAdvisorName': Or('BOHB'),
        'classArgs': {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('min_budget'): setNumberRange('min_budget', int, 0, 9999),
            Optional('max_budget'): setNumberRange('max_budget', int, 0, 9999),
            Optional('eta'):setNumberRange('eta', int, 0, 9999), 
            Optional('min_points_in_model'): setNumberRange('min_points_in_model', int, 0, 9999),
            Optional('top_n_percent'): setNumberRange('top_n_percent', int, 1, 99),
            Optional('num_samples'): setNumberRange('num_samples', int, 1, 9999),
            Optional('random_fraction'): setNumberRange('random_fraction', float, 0, 9999),
            Optional('bandwidth_factor'): setNumberRange('bandwidth_factor', float, 0, 9999),
            Optional('min_bandwidth'): setNumberRange('min_bandwidth', float, 0, 9999),
        },
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    'customized':{
        'codeDir': setPathCheck('codeDir'),
        'classFileName': setType('classFileName', str),
        'className': setType('className', str),
        Optional('classArgs'): dict,
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    }
}

assessor_schema_dict = {
    'Medianstop': {
        'builtinAssessorName': 'Medianstop',
        Optional('classArgs'): {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('start_step'): setNumberRange('start_step', int, 0, 9999),
        },
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    'Curvefitting': {
        'builtinAssessorName': 'Curvefitting',
        Optional('classArgs'): {
            'epoch_num': setNumberRange('epoch_num', int, 0, 9999),
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('start_step'): setNumberRange('start_step', int, 0, 9999),
            Optional('threshold'): setNumberRange('threshold', float, 0, 9999),
            Optional('gap'): setNumberRange('gap', int, 1, 9999),
        },
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    },
    'customized': {
        'codeDir': setPathCheck('codeDir'),
        'classFileName': setType('classFileName', str),
        'className': setType('className', str),
        Optional('classArgs'): dict,
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999)
    }
}

common_trial_schema = {
'trial':{
    'command': setType('command', str),
    'codeDir': setPathCheck('codeDir'),
    'gpuNum': setNumberRange('gpuNum', int, 0, 99999)
    }
}

pai_trial_schema = {
'trial':{
    'command': setType('command', str),
    'codeDir': setPathCheck('codeDir'),
    'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
    'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
    'memoryMB': setType('memoryMB', int),
    'image': setType('image', str),
    Optional('shmMB'): setType('shmMB', int),
    Optional('dataDir'): And(Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),\
                         error='ERROR: dataDir format error, dataDir format is hdfs://xxx.xxx.xxx.xxx:xxx'),
    Optional('outputDir'): And(Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),\
                         error='ERROR: outputDir format error, outputDir format is hdfs://xxx.xxx.xxx.xxx:xxx'),
    Optional('virtualCluster'): setType('virtualCluster', str),
    }
}

pai_config_schema = {
    'paiConfig':{
        'userName': setType('userName', str),
        'passWord': setType('passWord', str),
        'host': setType('host', str)
    }
}

kubeflow_trial_schema = {
'trial':{
        'codeDir':  setPathCheck('codeDir'),
        Optional('ps'): {
            'replicas': setType('replicas', int),
            'command': setType('command', str),
            'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
            'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
            'memoryMB': setType('memoryMB', int),
            'image': setType('image', str)
        },
        Optional('master'): {
            'replicas': setType('replicas', int),
            'command': setType('command', str),
            'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
            'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
            'memoryMB': setType('memoryMB', int),
            'image': setType('image', str)
        },
        Optional('worker'):{
            'replicas': setType('replicas', int),
            'command': setType('command', str),
            'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
            'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
            'memoryMB': setType('memoryMB', int),
            'image': setType('image', str)
        } 
    }
}

kubeflow_config_schema = {
    'kubeflowConfig':Or({
        'operator': setChoice('operator', 'tf-operator', 'pytorch-operator'),
        'apiVersion': setType('apiVersion', str),
        Optional('storage'): setChoice('storage', 'nfs', 'azureStorage'),
        'nfs': {
            'server': setType('server', str),
            'path': setType('path', str)
        }
    },{
        'operator': setChoice('operator', 'tf-operator', 'pytorch-operator'),
        'apiVersion': setType('apiVersion', str),
        Optional('storage'): setChoice('storage', 'nfs', 'azureStorage'),
        'keyVault': {
            'vaultName': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),\
                         error='ERROR: vaultName format error, vaultName support using (0-9|a-z|A-Z|-)'),
            'name': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),\
                    error='ERROR: name format error, name support using (0-9|a-z|A-Z|-)')
        },
        'azureStorage': {
            'accountName': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,31}'),\
                           error='ERROR: accountName format error, accountName support using (0-9|a-z|A-Z|-)'),
            'azureShare': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,63}'),\
                          error='ERROR: azureShare format error, azureShare support using (0-9|a-z|A-Z|-)')
        }
    })
}

frameworkcontroller_trial_schema = {
    'trial':{
        'codeDir':  setPathCheck('codeDir'),
        'taskRoles': [{
            'name': setType('name', str),
            'taskNum': setType('taskNum', int),
            'frameworkAttemptCompletionPolicy': {
                'minFailedTaskCount': setType('minFailedTaskCount', int),
                'minSucceededTaskCount': setType('minSucceededTaskCount', int),
            },
            'command': setType('command', str),
            'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
            'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
            'memoryMB': setType('memoryMB', int),
            'image': setType('image', str)
        }]
    }
}

frameworkcontroller_config_schema = {
    'frameworkcontrollerConfig':Or({
        Optional('storage'): setChoice('storage', 'nfs', 'azureStorage'),
        Optional('serviceAccountName'): setType('serviceAccountName', str),
        'nfs': {
            'server': setType('server', str),
            'path': setType('path', str)
        }
    },{
        Optional('storage'): setChoice('storage', 'nfs', 'azureStorage'),
        Optional('serviceAccountName'): setType('serviceAccountName', str),
        'keyVault': {
            'vaultName': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),\
                         error='ERROR: vaultName format error, vaultName support using (0-9|a-z|A-Z|-)'),
            'name': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),\
                    error='ERROR: name format error, name support using (0-9|a-z|A-Z|-)')
        },
        'azureStorage': {
            'accountName': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,31}'),\
                           error='ERROR: accountName format error, accountName support using (0-9|a-z|A-Z|-)'),
            'azureShare': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,63}'),\
                          error='ERROR: azureShare format error, azureShare support using (0-9|a-z|A-Z|-)')
        }
    })
}

machine_list_schima = {
Optional('machineList'):[Or({
    'ip': setType('ip', str),
    Optional('port'): setNumberRange('port', int, 1, 65535),
    'username': setType('username', str),
    'passwd': setType('passwd', str),
    Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!')
    },{
    'ip': setType('ip', str),
    Optional('port'): setNumberRange('port', int, 1, 65535),
    'username': setType('username', str),
    'sshKeyPath': setPathCheck('sshKeyPath'),
    Optional('passphrase'): setType('passphrase', str),
    Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!')
})]
}

LOCAL_CONFIG_SCHEMA = Schema({**common_schema, **common_trial_schema})

REMOTE_CONFIG_SCHEMA = Schema({**common_schema, **common_trial_schema, **machine_list_schima})

PAI_CONFIG_SCHEMA = Schema({**common_schema, **pai_trial_schema, **pai_config_schema})

KUBEFLOW_CONFIG_SCHEMA = Schema({**common_schema, **kubeflow_trial_schema, **kubeflow_config_schema})

FRAMEWORKCONTROLLER_CONFIG_SCHEMA = Schema({**common_schema, **frameworkcontroller_trial_schema, **frameworkcontroller_config_schema})
