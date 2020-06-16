# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from schema import Schema, And, Optional, Regex, Or
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
        'trainingServicePlatform', 'remote', 'local', 'pai', 'kubeflow', 'frameworkcontroller', 'paiYarn', 'dlts', 'aml'),
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
tuner_schema_dict = {
    'Anneal': {
        'builtinTunerName': 'Anneal',
        Optional('classArgs'): {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
        },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'SMAC': {
        'builtinTunerName': 'SMAC',
        Optional('classArgs'): {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('config_dedup'): setType('config_dedup', bool)
        },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    ('Evolution'): {
        'builtinTunerName': setChoice('builtinTunerName', 'Evolution'),
        Optional('classArgs'): {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('population_size'): setNumberRange('population_size', int, 0, 99999),
        },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    ('BatchTuner', 'GridSearch', 'Random'): {
        'builtinTunerName': setChoice('builtinTunerName', 'BatchTuner', 'GridSearch', 'Random'),
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'TPE': {
        'builtinTunerName': 'TPE',
        Optional('classArgs'): {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('parallel_optimize'): setType('parallel_optimize', bool),
            Optional('constant_liar_type'): setChoice('constant_liar_type', 'min', 'max', 'mean')
        },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'NetworkMorphism': {
        'builtinTunerName': 'NetworkMorphism',
        Optional('classArgs'): {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('task'): setChoice('task', 'cv', 'nlp', 'common'),
            Optional('input_width'): setType('input_width', int),
            Optional('input_channel'): setType('input_channel', int),
            Optional('n_output_node'): setType('n_output_node', int),
            },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'MetisTuner': {
        'builtinTunerName': 'MetisTuner',
        Optional('classArgs'): {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('no_resampling'): setType('no_resampling', bool),
            Optional('no_candidates'): setType('no_candidates', bool),
            Optional('selection_num_starting_points'):  setType('selection_num_starting_points', int),
            Optional('cold_start_num'): setType('cold_start_num', int),
            },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'GPTuner': {
        'builtinTunerName': 'GPTuner',
        Optional('classArgs'): {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('utility'): setChoice('utility', 'ei', 'ucb', 'poi'),
            Optional('kappa'): setType('kappa', float),
            Optional('xi'): setType('xi', float),
            Optional('nu'): setType('nu', float),
            Optional('alpha'): setType('alpha', float),
            Optional('cold_start_num'): setType('cold_start_num', int),
            Optional('selection_num_warm_up'):  setType('selection_num_warm_up', int),
            Optional('selection_num_starting_points'):  setType('selection_num_starting_points', int),
            },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'PPOTuner': {
        'builtinTunerName': 'PPOTuner',
        'classArgs': {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('trials_per_update'): setNumberRange('trials_per_update', int, 0, 99999),
            Optional('epochs_per_update'): setNumberRange('epochs_per_update', int, 0, 99999),
            Optional('minibatch_size'): setNumberRange('minibatch_size', int, 0, 99999),
            Optional('ent_coef'): setType('ent_coef', float),
            Optional('lr'): setType('lr', float),
            Optional('vf_coef'): setType('vf_coef', float),
            Optional('max_grad_norm'): setType('max_grad_norm', float),
            Optional('gamma'): setType('gamma', float),
            Optional('lam'): setType('lam', float),
            Optional('cliprange'): setType('cliprange', float),
        },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'PBTTuner': {
        'builtinTunerName': 'PBTTuner',
        'classArgs': {
            'optimize_mode': setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('all_checkpoint_dir'): setType('all_checkpoint_dir', str),
            Optional('population_size'): setNumberRange('population_size', int, 0, 99999),
            Optional('factors'): setType('factors', tuple),
            Optional('fraction'): setType('fraction', float),
        },
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'customized': {
        'codeDir': setPathCheck('codeDir'),
        'classFileName': setType('classFileName', str),
        'className': setType('className', str),
        Optional('classArgs'): dict,
        Optional('includeIntermediateResults'): setType('includeIntermediateResults', bool),
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
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
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
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
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    },
    'customized':{
        'codeDir': setPathCheck('codeDir'),
        'classFileName': setType('classFileName', str),
        'className': setType('className', str),
        Optional('classArgs'): dict,
        Optional('gpuIndices'): Or(int, And(str, lambda x: len([int(i) for i in x.split(',')]) > 0), error='gpuIndex format error!'),
    }
}

assessor_schema_dict = {
    'Medianstop': {
        'builtinAssessorName': 'Medianstop',
        Optional('classArgs'): {
            Optional('optimize_mode'): setChoice('optimize_mode', 'maximize', 'minimize'),
            Optional('start_step'): setNumberRange('start_step', int, 0, 9999),
        },
    },
    'Curvefitting': {
        'builtinAssessorName': 'Curvefitting',
        Optional('classArgs'): {
            'epoch_num': setNumberRange('epoch_num', int, 0, 9999),
            Optional('start_step'): setNumberRange('start_step', int, 0, 9999),
            Optional('threshold'): setNumberRange('threshold', float, 0, 9999),
            Optional('gap'): setNumberRange('gap', int, 1, 9999),
        },
    },
    'customized': {
        'codeDir': setPathCheck('codeDir'),
        'classFileName': setType('classFileName', str),
        'className': setType('className', str),
        Optional('classArgs'): dict,
    }
}

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
        Optional('paiStorageConfigName'): setType('paiStorageConfigName', str),
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

aml_trial_schema = {
    'trial':{
        'codeDir': setPathCheck('codeDir'),
        'script': setType('script', str),
        'image': setType('image', str),
        'computeTarget': setType('computeClusterName', str),
        'nodeCount': setType('nodeCount', int)
    }
}

aml_config_schema = {
    'amlConfig': {
        'subscriptionId': setType('subscriptionId', str),
        'resourceGroup': setType('resourceGroup', str),
        'workspaceName': setType('workspaceName', str),
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
