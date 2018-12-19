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

common_schema = {
'authorName': str,
'experimentName': str,
Optional('description'): str,
'trialConcurrency': And(int, lambda n: 1 <=n <= 999999),
Optional('maxExecDuration'): Regex(r'^[1-9][0-9]*[s|m|h|d]$'),
Optional('maxTrialNum'): And(int, lambda x: 1 <= x <= 99999),
'trainingServicePlatform': And(str, lambda x: x in ['remote', 'local', 'pai', 'kubeflow']),
Optional('searchSpacePath'): os.path.exists,
Optional('multiPhase'): bool,
Optional('multiThread'): bool,
Optional('nniManagerIp'): str,
'useAnnotation': bool,
Optional('advisor'): Or({
    'builtinAdvisorName': Or('Hyperband'),
    'classArgs': {
        'optimize_mode': Or('maximize', 'minimize'),
        Optional('R'): int,
        Optional('eta'): int
    },
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
},{
    'codeDir': os.path.exists,
    'classFileName': str,
    'className': str,
    Optional('classArgs'): dict,
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
}),
Optional('tuner'): Or({
    'builtinTunerName': Or('TPE', 'Random', 'Anneal', 'SMAC', 'Evolution'),
    Optional('classArgs'): {
        'optimize_mode': Or('maximize', 'minimize')
    },
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
},{
    'builtinTunerName': Or('BatchTuner', 'GridSearch'),
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
},{
    'builtinTunerName': 'NetworkMorphism',
    'classArgs': {
        Optional('optimize_mode'): Or('maximize', 'minimize'),
        Optional('task'): And(str, lambda x: x in ['cv','nlp','common']),
        Optional('input_width'):  int,
        Optional('input_channel'):  int,
        Optional('n_output_node'):  int,
        },
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
},{
    'codeDir': os.path.exists,
    'classFileName': str,
    'className': str,
    Optional('classArgs'): dict,
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
}),
Optional('assessor'): Or({
    'builtinAssessorName': lambda x: x in ['Medianstop'],
    Optional('classArgs'): {
        Optional('optimize_mode'): Or('maximize', 'minimize'),
        Optional('start_step'): And(int, lambda x: 0 <= x <= 9999)
    },
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999)
},{
    'codeDir': os.path.exists,
    'classFileName': str,
    'className': str,
    Optional('classArgs'): dict,
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
}),
}

common_trial_schema = {
'trial':{
    'command': str,
    'codeDir': os.path.exists,
    'gpuNum': And(int, lambda x: 0 <= x <= 99999)
    }
}

pai_trial_schema = {
'trial':{
    'command': str,
    'codeDir': os.path.exists,
    'gpuNum': And(int, lambda x: 0 <= x <= 99999),
    'cpuNum': And(int, lambda x: 0 <= x <= 99999),
    'memoryMB': int,
    'image': str,
    Optional('dataDir'): Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),
    Optional('outputDir'): Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),
    Optional('virtualCluster'): str
    }
}

pai_config_schema = {
'paiConfig':{
  'userName': str,
  'passWord': str,
  'host': str
}
}

kubeflow_trial_schema = {
'trial':{
        'codeDir':  os.path.exists,
        Optional('ps'): {
            'replicas': int,
            'command': str,
            'gpuNum': And(int, lambda x: 0 <= x <= 99999),
            'cpuNum': And(int, lambda x: 0 <= x <= 99999),
            'memoryMB': int,
            'image': str
        },
        Optional('master'): {
            'replicas': int,
            'command': str,
            'gpuNum': And(int, lambda x: 0 <= x <= 99999),
            'cpuNum': And(int, lambda x: 0 <= x <= 99999),
            'memoryMB': int,
            'image': str
        },
        Optional('worker'):{
            'replicas': int,
            'command': str,
            'gpuNum': And(int, lambda x: 0 <= x <= 99999),
            'cpuNum': And(int, lambda x: 0 <= x <= 99999),
            'memoryMB': int,
            'image': str
        } 
    }
}

kubeflow_config_schema = {
    'kubeflowConfig':Or({
        'operator': Or('tf-operator', 'pytorch-operator'),
        'apiVersion': str,
        Optional('storage'): Or('nfs', 'azureStorage'),
        'nfs': {
            'server': str,
            'path': str
        }
    },{
        'operator': Or('tf-operator', 'pytorch-operator'),
        'apiVersion': str,
        Optional('storage'): Or('nfs', 'azureStorage'),
        'keyVault': {
            'vaultName': Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),
            'name': Regex('([0-9]|[a-z]|[A-Z]|-){1,127}')
        },
        'azureStorage': {
            'accountName': Regex('([0-9]|[a-z]|[A-Z]|-){3,31}'),
            'azureShare': Regex('([0-9]|[a-z]|[A-Z]|-){3,63}')
        }
    })
}

machine_list_schima = {
Optional('machineList'):[Or({
    'ip': str,
    Optional('port'): And(int, lambda x: 0 < x < 65535),
    'username': str,
    'passwd': str
    },{
    'ip': str,
    Optional('port'): And(int, lambda x: 0 < x < 65535),
    'username': str,
    'sshKeyPath': os.path.exists,
    Optional('passphrase'): str
})]
}

LOCAL_CONFIG_SCHEMA = Schema({**common_schema, **common_trial_schema})

REMOTE_CONFIG_SCHEMA = Schema({**common_schema, **common_trial_schema, **machine_list_schima})

PAI_CONFIG_SCHEMA = Schema({**common_schema, **pai_trial_schema, **pai_config_schema})

KUBEFLOW_CONFIG_SCHEMA = Schema({**common_schema, **kubeflow_trial_schema, **kubeflow_config_schema})
