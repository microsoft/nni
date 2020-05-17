# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from schema import Schema, And, Optional, Regex, Or, SchemaError
from nni.package_utils import create_validator_instance, get_all_builtin_names, get_builtin_algo_meta
from .constants import SCHEMA_TYPE_ERROR, SCHEMA_RANGE_ERROR, SCHEMA_PATH_ERROR


def setType(key, valueType):
    '''check key type'''
    return And(valueType, error=SCHEMA_TYPE_ERROR % (key, valueType.__name__))

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
    Optional('maxExecDuration'): And(Regex(r'^[1-9][0-9]*[s|m|h|d]$', error='ERROR: maxExecDuration format is [digit]{s,m,h,d}')),
    Optional('maxTrialNum'): setNumberRange('maxTrialNum', int, 1, 99999),
    'trainingServicePlatform': setChoice(
        'trainingServicePlatform', 'remote', 'local', 'pai', 'kubeflow', 'frameworkcontroller', 'paiYarn', 'dlts'),
    Optional('searchSpacePath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'searchSpacePath'),
    Optional('multiPhase'): setType('multiPhase', bool),
    Optional('multiThread'): setType('multiThread', bool),
    Optional('nniManagerIp'): setType('nniManagerIp', str),
    Optional('logDir'): And(os.path.isdir, error=SCHEMA_PATH_ERROR % 'logDir'),
    Optional('debug'): setType('debug', bool),
    Optional('versionCheck'): setType('versionCheck', bool),
    Optional('logLevel'): setChoice('logLevel', 'trace', 'debug', 'info', 'warning', 'error', 'fatal'),
    Optional('logCollection'): setChoice('logCollection', 'http', 'none'),
    'useAnnotation': setType('useAnnotation', bool),
    Optional('tuner'): dict,
    Optional('advisor'): dict,
    Optional('assessor'): dict,
    Optional('localConfig'): {
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
        Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
        Optional('useActiveGpu'): setType('useActiveGpu', bool)
    }
}

class AlgoSchema:
    def __init__(self):
        self.algo_schema = {
            Optional('codeDir'): setPathCheck('codeDir'),
            Optional('classFileName'): setType('classFileName', str),
            Optional('className'): setType('className', str),
            Optional('classArgs'): dict,
            Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
            Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
        }
        self.builtin_keys = {
            'tuner': 'builtinTunerName',
            'assessor': 'builtinAssessorName',
            'advisor': 'builtinAdvisorName'
        }
        self.builtin_name_schema = {}
        for k, n in self.builtin_keys.items():
            self.builtin_name_schema[k] = {Optional(n): setChoice(n, *get_all_builtin_names(k+'s'))}

        self.customized_keys = set(['codeDir', 'classFileName', 'className'])

    def validate_class_args(self, class_args, algo_type, builtin_name):
        if not builtin_name or not class_args:
            return
        meta = get_builtin_algo_meta(algo_type+'s', builtin_name)
        if meta and 'accept_class_args' in meta and meta['accept_class_args'] == False:
            raise SchemaError('classArgs is not allowed.')

        validator = create_validator_instance(algo_type+'s', builtin_name)
        if validator:
            try:
                print('validating:', validator, class_args)
                validator.validate_class_args(**class_args)
            except Exception as e:
                raise SchemaError(str(e))

    def missing_customized_keys(self, data):
        return self.customized_keys - set(data.keys())

    def validate_extras(self, data, algo_type):
        builtin_key = self.builtin_keys[algo_type]
        if (builtin_key in data) and (set(data.keys()) & self.customized_keys):
            raise SchemaError('{} and {} cannot be specified at the same time.'.format(
                builtin_key, set(data.keys()) & self.customized_keys
            ))

        if self.missing_customized_keys(data) and builtin_key not in data:
            raise SchemaError('Either customized {} ({}) or builtin {} ({}) must be set.'.format(
                algo_type, self.customized_keys, algo_type, builtin_key))

        if not self.missing_customized_keys(data):
            class_file_name = os.path.join(data['codeDir'], data['classFileName'])
            if not os.path.isfile(class_file_name):
                raise SchemaError('classFileName {} not found.'.format(class_file_name))

        builtin_name = data.get(builtin_key)
        class_args = data.get('classArgs')
        self.validate_class_args(class_args, algo_type, builtin_name)

    def validate(self, data, algo_type):
        assert algo_type in ['tuner', 'assessor', 'advisor']
        self.algo_schema.update(self.builtin_name_schema[algo_type])
        Schema(self.algo_schema).validate(data)
        self.validate_extras(data, algo_type)


common_trial_schema = {
    'trial':{
        'command': setType('command', str),
        'codeDir': setPathCheck('codeDir'),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
        Optional('nasMode'): setChoice('nasMode', 'classic_mode', 'enas_mode', 'oneshot_mode', 'darts_mode')
    }
}

pai_yarn_trial_schema = {
    'trial':{
        'command': setType('command', str),
        'codeDir': setPathCheck('codeDir'),
        'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
        'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
        'memoryMB': setType('memoryMB', int),
        'image': setType('image', str),
        Optional('authFile'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'authFile'),
        Optional('shmMB'): setType('shmMB', int),
        Optional('dataDir'): And(Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),\
                            error='ERROR: dataDir format error, dataDir format is hdfs://xxx.xxx.xxx.xxx:xxx'),
        Optional('outputDir'): And(Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),\
                            error='ERROR: outputDir format error, outputDir format is hdfs://xxx.xxx.xxx.xxx:xxx'),
        Optional('virtualCluster'): setType('virtualCluster', str),
        Optional('nasMode'): setChoice('nasMode', 'classic_mode', 'enas_mode', 'oneshot_mode', 'darts_mode'),
        Optional('portList'): [{
            "label": setType('label', str),
            "beginAt": setType('beginAt', int),
            "portNumber": setType('portNumber', int)
        }]
    }
}

pai_yarn_config_schema = {
    'paiYarnConfig': Or({
        'userName': setType('userName', str),
        'passWord': setType('passWord', str),
        'host': setType('host', str)
    }, {
        'userName': setType('userName', str),
        'token': setType('token', str),
        'host': setType('host', str)
    })
}


pai_trial_schema = {
    'trial':{
        'codeDir': setPathCheck('codeDir'),
        'nniManagerNFSMountPath': setPathCheck('nniManagerNFSMountPath'),
        'containerNFSMountPath': setType('containerNFSMountPath', str),
        Optional('command'): setType('command', str),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
        Optional('cpuNum'): setNumberRange('cpuNum', int, 0, 99999),
        Optional('memoryMB'): setType('memoryMB', int),
        Optional('image'): setType('image', str),
        Optional('virtualCluster'): setType('virtualCluster', str),
        Optional('paiStoragePlugin'): setType('paiStoragePlugin', str),
        Optional('paiConfigPath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'paiConfigPath')
    }
}

pai_config_schema = {
    'paiConfig': Or({
        'userName': setType('userName', str),
        'passWord': setType('passWord', str),
        'host': setType('host', str)
    }, {
        'userName': setType('userName', str),
        'token': setType('token', str),
        'host': setType('host', str)
    })
}

dlts_trial_schema = {
    'trial':{
        'command': setType('command', str),
        'codeDir': setPathCheck('codeDir'),
        'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
        'image': setType('image', str),
    }
}

dlts_config_schema = {
    'dltsConfig': {
        'dashboard': setType('dashboard', str),

        Optional('cluster'): setType('cluster', str),
        Optional('team'): setType('team', str),

        Optional('email'): setType('email', str),
        Optional('password'): setType('password', str),
    }
}

kubeflow_trial_schema = {
    'trial':{
        'codeDir':  setPathCheck('codeDir'),
        Optional('nasMode'): setChoice('nasMode', 'classic_mode', 'enas_mode', 'oneshot_mode', 'darts_mode'),
        Optional('ps'): {
            'replicas': setType('replicas', int),
            'command': setType('command', str),
            'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
            'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
            'memoryMB': setType('memoryMB', int),
            'image': setType('image', str),
            Optional('privateRegistryAuthPath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'privateRegistryAuthPath')
        },
        Optional('master'): {
            'replicas': setType('replicas', int),
            'command': setType('command', str),
            'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
            'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
            'memoryMB': setType('memoryMB', int),
            'image': setType('image', str),
            Optional('privateRegistryAuthPath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'privateRegistryAuthPath')
        },
        Optional('worker'):{
            'replicas': setType('replicas', int),
            'command': setType('command', str),
            'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
            'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
            'memoryMB': setType('memoryMB', int),
            'image': setType('image', str),
            Optional('privateRegistryAuthPath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'privateRegistryAuthPath')
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
    }, {
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
        },
        Optional('uploadRetryCount'): setNumberRange('uploadRetryCount', int, 1, 99999)
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
            'image': setType('image', str),
            Optional('privateRegistryAuthPath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'privateRegistryAuthPath')
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
    }, {
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
        },
        Optional('uploadRetryCount'): setNumberRange('uploadRetryCount', int, 1, 99999)
    })
}

machine_list_schema = {
    Optional('machineList'):[Or(
        {
            'ip': setType('ip', str),
            Optional('port'): setNumberRange('port', int, 1, 65535),
            'username': setType('username', str),
            'sshKeyPath': setPathCheck('sshKeyPath'),
            Optional('passphrase'): setType('passphrase', str),
            Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
            Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
            Optional('useActiveGpu'): setType('useActiveGpu', bool)
        },
        {
            'ip': setType('ip', str),
            Optional('port'): setNumberRange('port', int, 1, 65535),
            'username': setType('username', str),
            'passwd': setType('passwd', str),
            Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
            Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
            Optional('useActiveGpu'): setType('useActiveGpu', bool)
        })]
}

LOCAL_CONFIG_SCHEMA = Schema({**common_schema, **common_trial_schema})

REMOTE_CONFIG_SCHEMA = Schema({**common_schema, **common_trial_schema, **machine_list_schema})

PAI_CONFIG_SCHEMA = Schema({**common_schema, **pai_trial_schema, **pai_config_schema})

PAI_YARN_CONFIG_SCHEMA = Schema({**common_schema, **pai_yarn_trial_schema, **pai_yarn_config_schema})

DLTS_CONFIG_SCHEMA = Schema({**common_schema, **dlts_trial_schema, **dlts_config_schema})

KUBEFLOW_CONFIG_SCHEMA = Schema({**common_schema, **kubeflow_trial_schema, **kubeflow_config_schema})

FRAMEWORKCONTROLLER_CONFIG_SCHEMA = Schema({**common_schema, **frameworkcontroller_trial_schema, **frameworkcontroller_config_schema})
