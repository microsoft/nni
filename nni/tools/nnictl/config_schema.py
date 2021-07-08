# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os

import netifaces
from schema import And, Optional, Or, Regex, Schema, SchemaError
from nni.tools.package_utils import (
    create_validator_instance,
    get_all_builtin_names,
    get_registered_algo_meta,
)

from .common_utils import get_yml_content, print_warning
from .constants import SCHEMA_PATH_ERROR, SCHEMA_RANGE_ERROR, SCHEMA_TYPE_ERROR


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


class AlgoSchema:
    """
    This class is the schema of 'tuner', 'assessor' and 'advisor' sections of experiment configuraion file.
    For example:
    AlgoSchema('tuner') creates the schema of tuner section.
    """

    def __init__(self, algo_type):
        """
        Parameters:
        -----------
        algo_type: str
            One of ['tuner', 'assessor', 'advisor'].
            'tuner': This AlgoSchema class create the schema of tuner section.
            'assessor': This AlgoSchema class create the schema of assessor section.
            'advisor': This AlgoSchema class create the schema of advisor section.
        """
        assert algo_type in ['tuner', 'assessor', 'advisor']
        self.algo_type = algo_type
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
        meta = get_registered_algo_meta(builtin_name, algo_type+'s')
        if meta and 'acceptClassArgs' in meta and meta['acceptClassArgs'] == False:
            raise SchemaError('classArgs is not allowed.')

        logging.getLogger('nni.protocol').setLevel(logging.ERROR)  # we know IPC is not there, don't complain
        validator = create_validator_instance(algo_type+'s', builtin_name)
        if validator:
            try:
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

    def validate(self, data):
        self.algo_schema.update(self.builtin_name_schema[self.algo_type])
        Schema(self.algo_schema).validate(data)
        self.validate_extras(data, self.algo_type)


common_schema = {
    'authorName': setType('authorName', str),
    'experimentName': setType('experimentName', str),
    Optional('description'): setType('description', str),
    'trialConcurrency': setNumberRange('trialConcurrency', int, 1, 99999),
    Optional('maxExecDuration'): And(Regex(r'^[1-9][0-9]*[s|m|h|d]$', error='ERROR: maxExecDuration format is [digit]{s,m,h,d}')),
    Optional('maxTrialNum'): setNumberRange('maxTrialNum', int, 1, 99999),
    'trainingServicePlatform': setChoice(
        'trainingServicePlatform', 'remote', 'local', 'pai', 'kubeflow', 'frameworkcontroller', 'dlts', 'aml', 'adl', 'hybrid'),
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
    Optional('tuner'): AlgoSchema('tuner'),
    Optional('advisor'): AlgoSchema('advisor'),
    Optional('assessor'): AlgoSchema('assessor'),
    Optional('localConfig'): {
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
        Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
        Optional('useActiveGpu'): setType('useActiveGpu', bool)
    },
    Optional('sharedStorage'): {
        'storageType': setChoice('storageType', 'NFS', 'AzureBlob'),
        Optional('localMountPoint'): setType('localMountPoint', str),
        Optional('remoteMountPoint'): setType('remoteMountPoint', str),
        Optional('nfsServer'): setType('nfsServer', str),
        Optional('exportedDirectory'): setType('exportedDirectory', str),
        Optional('storageAccountName'): setType('storageAccountName', str),
        Optional('storageAccountKey'): setType('storageAccountKey', str),
        Optional('containerName'): setType('containerName', str),
        Optional('resourceGroupName'): setType('resourceGroupName', str),
        Optional('localMounted'): setChoice('localMounted', 'usermount', 'nnimount', 'nomount')
    }
}

common_trial_schema = {
    'trial': {
        'command': setType('command', str),
        'codeDir': setPathCheck('codeDir'),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
        Optional('nasMode'): setChoice('nasMode', 'classic_mode', 'enas_mode', 'oneshot_mode', 'darts_mode')
    }
}

pai_yarn_trial_schema = {
    'trial': {
        'command': setType('command', str),
        'codeDir': setPathCheck('codeDir'),
        'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
        'cpuNum': setNumberRange('cpuNum', int, 0, 99999),
        'memoryMB': setType('memoryMB', int),
        'image': setType('image', str),
        Optional('authFile'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'authFile'),
        Optional('shmMB'): setType('shmMB', int),
        Optional('dataDir'): And(Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),
                                 error='ERROR: dataDir format error, dataDir format is hdfs://xxx.xxx.xxx.xxx:xxx'),
        Optional('outputDir'): And(Regex(r'hdfs://(([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(/.*)?'),
                                   error='ERROR: outputDir format error, outputDir format is hdfs://xxx.xxx.xxx.xxx:xxx'),
        Optional('virtualCluster'): setType('virtualCluster', str),
        Optional('nasMode'): setChoice('nasMode', 'classic_mode', 'enas_mode', 'oneshot_mode', 'darts_mode'),
        Optional('portList'): [{
            'label': setType('label', str),
            'beginAt': setType('beginAt', int),
            'portNumber': setType('portNumber', int)
        }]
    }
}


pai_trial_schema = {
    'trial': {
        'codeDir': setPathCheck('codeDir'),
        'nniManagerNFSMountPath': setPathCheck('nniManagerNFSMountPath'),
        'containerNFSMountPath': setType('containerNFSMountPath', str),
        Optional('command'): setType('command', str),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
        Optional('cpuNum'): setNumberRange('cpuNum', int, 0, 99999),
        Optional('memoryMB'): setType('memoryMB', int),
        Optional('image'): setType('image', str),
        Optional('virtualCluster'): setType('virtualCluster', str),
        Optional('paiStorageConfigName'): setType('paiStorageConfigName', str),
        Optional('paiConfigPath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'paiConfigPath')
    }
}

pai_config_schema = {
    Optional('paiConfig'): {
        'userName': setType('userName', str),
        Or('passWord', 'token', only_one=True): str,
        'host': setType('host', str),
        Optional('reuse'): setType('reuse', bool),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
        Optional('cpuNum'): setNumberRange('cpuNum', int, 0, 99999),
        Optional('memoryMB'): setType('memoryMB', int),
        Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
        Optional('useActiveGpu'): setType('useActiveGpu', bool),
    }
}

dlts_trial_schema = {
    'trial': {
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

aml_trial_schema = {
    'trial': {
        'codeDir': setPathCheck('codeDir'),
        'command': setType('command', str),
        'image': setType('image', str),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
    }
}

aml_config_schema = {
    Optional('amlConfig'): {
        'subscriptionId': setType('subscriptionId', str),
        'resourceGroup': setType('resourceGroup', str),
        'workspaceName': setType('workspaceName', str),
        'computeTarget': setType('computeTarget', str),
        Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
        Optional('useActiveGpu'): setType('useActiveGpu', bool),
    }
}

hybrid_trial_schema = {
    'trial': {
        'codeDir': setPathCheck('codeDir'),
        Optional('nniManagerNFSMountPath'): setPathCheck('nniManagerNFSMountPath'),
        Optional('containerNFSMountPath'): setType('containerNFSMountPath', str),
        Optional('nasMode'): setChoice('nasMode', 'classic_mode', 'enas_mode', 'oneshot_mode', 'darts_mode'),
        'command': setType('command', str),
        Optional('gpuNum'): setNumberRange('gpuNum', int, 0, 99999),
        Optional('cpuNum'): setNumberRange('cpuNum', int, 0, 99999),
        Optional('memoryMB'): setType('memoryMB', int),
        Optional('image'): setType('image', str),
        Optional('virtualCluster'): setType('virtualCluster', str),
        Optional('paiStorageConfigName'): setType('paiStorageConfigName', str),
        Optional('paiConfigPath'): And(os.path.exists, error=SCHEMA_PATH_ERROR % 'paiConfigPath')
    }
}

hybrid_config_schema = {
    'hybridConfig': {
        'trainingServicePlatforms': ['local', 'remote', 'pai', 'aml']
    }
}

adl_trial_schema = {
    'trial':{
        'codeDir': setType('codeDir', str),
        'command': setType('command', str),
        'gpuNum': setNumberRange('gpuNum', int, 0, 99999),
        'image': setType('image', str),
        Optional('namespace'): setType('namespace', str),
        Optional('imagePullSecrets'): [{
            'name': setType('name', str)
        }],
        Optional('nfs'): {
            'server': setType('server', str),
            'path': setType('path', str),
            'containerMountPath': setType('containerMountPath', str)
        },
        Optional('adaptive'): setType('adaptive', bool),
        Optional('checkpoint'): {
            'storageClass': setType('storageClass', str),
            'storageSize': setType('storageSize', str)
        },
        Optional('cpuNum'): setNumberRange('cpuNum', int, 0, 99999),
        Optional('memorySize'): setType('memorySize', str)
    }
}

kubeflow_trial_schema = {
    'trial': {
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
        Optional('worker'): {
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
    'kubeflowConfig': Or({
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
            'vaultName': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),
                             error='ERROR: vaultName format error, vaultName support using (0-9|a-z|A-Z|-)'),
            'name': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),
                        error='ERROR: name format error, name support using (0-9|a-z|A-Z|-)')
        },
        'azureStorage': {
            'accountName': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,31}'),
                               error='ERROR: accountName format error, accountName support using (0-9|a-z|A-Z|-)'),
            'azureShare': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,63}'),
                              error='ERROR: azureShare format error, azureShare support using (0-9|a-z|A-Z|-)')
        },
        Optional('uploadRetryCount'): setNumberRange('uploadRetryCount', int, 1, 99999)
    })
}

frameworkcontroller_trial_schema = {
    'trial': {
        'codeDir':  setPathCheck('codeDir'),
        Optional('taskRoles'): [{
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
    'frameworkcontrollerConfig': Or({
        Optional('storage'): setChoice('storage', 'nfs', 'azureStorage', 'pvc'),
        Optional('serviceAccountName'): setType('serviceAccountName', str),
        'nfs': {
            'server': setType('server', str),
            'path': setType('path', str)
        },
        Optional('namespace'): setType('namespace', str),
        Optional('configPath'): setType('configPath', str),
    }, {
        Optional('storage'): setChoice('storage', 'nfs', 'azureStorage', 'pvc'),
        Optional('serviceAccountName'): setType('serviceAccountName', str),
        'configPath': setType('configPath', str),
        'pvc': {'path': setType('server', str)},
        Optional('namespace'): setType('namespace', str),
    }, {
        Optional('storage'): setChoice('storage', 'nfs', 'azureStorage', 'pvc'),
        Optional('serviceAccountName'): setType('serviceAccountName', str),
        'keyVault': {
            'vaultName': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),
                             error='ERROR: vaultName format error, vaultName support using (0-9|a-z|A-Z|-)'),
            'name': And(Regex('([0-9]|[a-z]|[A-Z]|-){1,127}'),
                        error='ERROR: name format error, name support using (0-9|a-z|A-Z|-)')
        },
        'azureStorage': {
            'accountName': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,31}'),
                               error='ERROR: accountName format error, accountName support using (0-9|a-z|A-Z|-)'),
            'azureShare': And(Regex('([0-9]|[a-z]|[A-Z]|-){3,63}'),
                              error='ERROR: azureShare format error, azureShare support using (0-9|a-z|A-Z|-)')
        },
        Optional('uploadRetryCount'): setNumberRange('uploadRetryCount', int, 1, 99999),
        Optional('namespace'): setType('namespace', str),
        Optional('configPath'): setType('configPath', str),
    })
}

remote_config_schema = {
    Optional('remoteConfig'): {
        'reuse': setType('reuse', bool)
    }
}

machine_list_schema = {
    Optional('machineList'): [Or(
        {
            'ip': setType('ip', str),
            Optional('port'): setNumberRange('port', int, 1, 65535),
            'username': setType('username', str),
            'sshKeyPath': setPathCheck('sshKeyPath'),
            Optional('passphrase'): setType('passphrase', str),
            Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
            Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
            Optional('useActiveGpu'): setType('useActiveGpu', bool),
            Optional('pythonPath'): setType('pythonPath', str)
        },
        {
            'ip': setType('ip', str),
            Optional('port'): setNumberRange('port', int, 1, 65535),
            'username': setType('username', str),
            'passwd': setType('passwd', str),
            Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
            Optional('maxTrialNumPerGpu'): setType('maxTrialNumPerGpu', int),
            Optional('useActiveGpu'): setType('useActiveGpu', bool),
            Optional('pythonPath'): setType('pythonPath', str)
        })]
}

training_service_schema_dict = {
    'adl': Schema({**common_schema, **adl_trial_schema}),
    'local': Schema({**common_schema, **common_trial_schema}),
    'remote': Schema({**common_schema, **common_trial_schema, **machine_list_schema, **remote_config_schema}),
    'pai': Schema({**common_schema, **pai_trial_schema, **pai_config_schema}),
    'kubeflow': Schema({**common_schema, **kubeflow_trial_schema, **kubeflow_config_schema}),
    'frameworkcontroller': Schema({**common_schema, **frameworkcontroller_trial_schema, **frameworkcontroller_config_schema}),
    'aml': Schema({**common_schema, **aml_trial_schema, **aml_config_schema}),
    'dlts': Schema({**common_schema, **dlts_trial_schema, **dlts_config_schema}),
    'hybrid': Schema({**common_schema, **hybrid_trial_schema, **hybrid_config_schema, **machine_list_schema,
                             **pai_config_schema, **aml_config_schema, **remote_config_schema}),
}


class NNIConfigSchema:
    def validate(self, data):
        train_service = data['trainingServicePlatform']
        Schema(common_schema['trainingServicePlatform']).validate(train_service)
        train_service_schema = training_service_schema_dict[train_service]
        train_service_schema.validate(data)
        self.validate_extras(data)

    def validate_extras(self, experiment_config):
        self.validate_tuner_adivosr_assessor(experiment_config)
        self.validate_pai_trial_conifg(experiment_config)
        self.validate_kubeflow_operators(experiment_config)
        self.validate_eth0_device(experiment_config)
        self.validate_hybrid_platforms(experiment_config)
        self.validate_frameworkcontroller_trial_config(experiment_config)

    def validate_tuner_adivosr_assessor(self, experiment_config):
        if experiment_config.get('advisor'):
            if experiment_config.get('assessor') or experiment_config.get('tuner'):
                raise SchemaError('advisor could not be set with assessor or tuner simultaneously!')
            self.validate_annotation_content(experiment_config, 'advisor', 'builtinAdvisorName')
        else:
            if not experiment_config.get('tuner'):
                raise SchemaError('Please provide tuner spec!')
            self.validate_annotation_content(experiment_config, 'tuner', 'builtinTunerName')

    def validate_search_space_content(self, experiment_config):
        '''Validate searchspace content,
        if the searchspace file is not json format or its values does not contain _type and _value which must be specified,
        it will not be a valid searchspace file'''
        try:
            search_space_content = json.load(open(experiment_config.get('searchSpacePath'), 'r'))
            for value in search_space_content.values():
                if not value.get('_type') or not value.get('_value'):
                    raise SchemaError('please use _type and _value to specify searchspace!')
        except Exception as e:
            raise SchemaError('searchspace file is not a valid json format! ' + str(e))

    def validate_kubeflow_operators(self, experiment_config):
        '''Validate whether the kubeflow operators are valid'''
        if experiment_config.get('kubeflowConfig'):
            if experiment_config.get('kubeflowConfig').get('operator') == 'tf-operator':
                if experiment_config.get('trial').get('master') is not None:
                    raise SchemaError('kubeflow with tf-operator can not set master')
                if experiment_config.get('trial').get('worker') is None:
                    raise SchemaError('kubeflow with tf-operator must set worker')
            elif experiment_config.get('kubeflowConfig').get('operator') == 'pytorch-operator':
                if experiment_config.get('trial').get('ps') is not None:
                    raise SchemaError('kubeflow with pytorch-operator can not set ps')
                if experiment_config.get('trial').get('master') is None:
                    raise SchemaError('kubeflow with pytorch-operator must set master')

            if experiment_config.get('kubeflowConfig').get('storage') == 'nfs':
                if experiment_config.get('kubeflowConfig').get('nfs') is None:
                    raise SchemaError('please set nfs configuration!')
            elif experiment_config.get('kubeflowConfig').get('storage') == 'azureStorage':
                if experiment_config.get('kubeflowConfig').get('azureStorage') is None:
                    raise SchemaError('please set azureStorage configuration!')
            elif experiment_config.get('kubeflowConfig').get('storage') is None:
                if experiment_config.get('kubeflowConfig').get('azureStorage'):
                    raise SchemaError('please set storage type!')

    def validate_annotation_content(self, experiment_config, spec_key, builtin_name):
        '''
        Valid whether useAnnotation and searchSpacePath is coexist
        spec_key: 'advisor' or 'tuner'
        builtin_name: 'builtinAdvisorName' or 'builtinTunerName'
        '''
        if experiment_config.get('useAnnotation'):
            if experiment_config.get('searchSpacePath'):
                raise SchemaError('If you set useAnnotation=true, please leave searchSpacePath empty')
        else:
            # validate searchSpaceFile
            if experiment_config[spec_key].get(builtin_name) == 'NetworkMorphism':
                return
            if experiment_config[spec_key].get(builtin_name):
                if experiment_config.get('searchSpacePath') is None:
                    raise SchemaError('Please set searchSpacePath!')
                self.validate_search_space_content(experiment_config)

    def validate_pai_config_path(self, experiment_config):
        '''validate paiConfigPath field'''
        if experiment_config.get('trainingServicePlatform') == 'pai':
            if experiment_config.get('trial', {}).get('paiConfigPath'):
                # validate commands
                pai_config = get_yml_content(experiment_config['trial']['paiConfigPath'])
                taskRoles_dict = pai_config.get('taskRoles')
                if not taskRoles_dict:
                    raise SchemaError('Please set taskRoles in paiConfigPath config file!')
            else:
                pai_trial_fields_required_list = ['image', 'paiStorageConfigName', 'command']
                for trial_field in pai_trial_fields_required_list:
                    if experiment_config['trial'].get(trial_field) is None:
                        raise SchemaError('Please set {0} in trial configuration,\
                                    or set additional pai configuration file path in paiConfigPath!'.format(trial_field))
                pai_resource_fields_required_list = ['gpuNum', 'cpuNum', 'memoryMB']
                for required_field in pai_resource_fields_required_list:
                    if experiment_config['trial'].get(required_field) is None and \
                            experiment_config['paiConfig'].get(required_field) is None:
                        raise SchemaError('Please set {0} in trial or paiConfig configuration,\
                                    or set additional pai configuration file path in paiConfigPath!'.format(required_field))

    def validate_pai_trial_conifg(self, experiment_config):
        '''validate the trial config in pai platform'''
        if experiment_config.get('trainingServicePlatform') in ['pai']:
            if experiment_config.get('trial').get('shmMB') and \
                    experiment_config['trial']['shmMB'] > experiment_config['trial']['memoryMB']:
                raise SchemaError('shmMB should be no more than memoryMB!')
            # backward compatibility
            warning_information = '{0} is not supported in NNI anymore, please remove the field in config file!\
            please refer https://github.com/microsoft/nni/blob/master/docs/en_US/TrainingService/PaiMode.md#run-an-experiment\
            for the practices of how to get data and output model in trial code'
            if experiment_config.get('trial').get('dataDir'):
                print_warning(warning_information.format('dataDir'))
            if experiment_config.get('trial').get('outputDir'):
                print_warning(warning_information.format('outputDir'))
            self.validate_pai_config_path(experiment_config)

    def validate_eth0_device(self, experiment_config):
        '''validate whether the machine has eth0 device'''
        if experiment_config.get('trainingServicePlatform') not in ['local'] \
                and not experiment_config.get('nniManagerIp') \
                and 'eth0' not in netifaces.interfaces():
            raise SchemaError('This machine does not contain eth0 network device, please set nniManagerIp in config file!')

    def validate_hybrid_platforms(self, experiment_config):
        required_config_name_map = {
            'remote': 'machineList',
            'aml': 'amlConfig',
            'pai': 'paiConfig'
        }
        if experiment_config.get('trainingServicePlatform') == 'hybrid':
            for platform in experiment_config['hybridConfig']['trainingServicePlatforms']:
                config_name = required_config_name_map.get(platform)
                if config_name and not experiment_config.get(config_name):
                    raise SchemaError('Need to set {0} for {1} in hybrid mode!'.format(config_name, platform))

    def validate_frameworkcontroller_trial_config(self, experiment_config):
        if experiment_config.get('trainingServicePlatform') == 'frameworkcontroller':
            if not experiment_config.get('trial').get('taskRoles'):
                if not experiment_config.get('frameworkcontrollerConfig').get('configPath'):
                    raise SchemaError("""If no taskRoles are specified a valid custom frameworkcontroller config should
                                         be set using the configPath attribute in frameworkcontrollerConfig!""")
                config_content = get_yml_content(experiment_config.get('frameworkcontrollerConfig').get('configPath'))
                if not config_content.get('spec').get('taskRoles') or not config_content.get('spec').get('taskRoles'):
                    raise SchemaError('Invalid frameworkcontroller config! No taskRoles were specified!')
                if not config_content.get('spec').get('taskRoles')[0].get('task'):
                    raise SchemaError('Invalid frameworkcontroller config! No task was specified for taskRole!')
                names = []
                for taskRole in config_content.get('spec').get('taskRoles'):
                    if not "name" in taskRole:
                        raise SchemaError('Invalid frameworkcontroller config! Name is missing for taskRole!')
                    names.append(taskRole.get("name"))
                if len(names) > len(set(names)):
                    raise SchemaError('Invalid frameworkcontroller config! Duplicate taskrole names!')
                if not config_content.get('metadata').get('name'):
                    raise SchemaError('Invalid frameworkcontroller config! No experiment name was specified!')

