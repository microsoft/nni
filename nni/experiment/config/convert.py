# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging

_logger = logging.getLogger(__name__)

def to_v2(v1):
    v1 = copy.deepcopy(v1)
    platform = v1.pop('trainingServicePlatform')
    assert platform in ['local', 'remote', 'pai', 'aml', 'kubeflow', 'frameworkcontroller']
    if platform == 'pai':
        platform = 'openpai'

    v2 = {}

    _drop_field(v1, 'authorName')
    _move_field(v1, v2, 'experimentName')
    _drop_field(v1, 'description')
    _move_field(v1, v2, 'trialConcurrency')
    _move_field(v1, v2, 'maxExecDuration', 'maxExperimentDuration')
    _move_field(v1, v2, 'maxTrialNum', 'maxTrialNumber')
    _move_field(v1, v2, 'searchSpacePath', 'searchSpaceFile')
    assert not v1.pop('multiPhase', None), 'Multi-phase is no longer supported'
    _deprecate(v1, v2, 'multiThread')
    _move_field(v1, v2, 'nniManagerIp')
    _move_field(v1, v2, 'logDir', 'experimentWorkingDirectory')
    _move_field(v1, v2, 'debug')
    _deprecate(v1, v2, 'versionCheck')
    _move_field(v1, v2, 'logLevel')
    _deprecate(v1, v2, 'logCollection')
    _move_field(v1, v2, 'useAnnotation')

    if 'trial' in v1:
        v1_trial = v1.pop('trial')
        _move_field(v1_trial, v2, 'command', 'trialCommand')
        _move_field(v1_trial, v2, 'codeDir', 'trialCodeDirectory')
        _move_field(v1_trial, v2, 'gpuNum', 'trialGpuNumber')

    for algo_type in ['tuner', 'assessor', 'advisor']:
        v1_algo = v1.pop(algo_type, None)
        if not v1_algo:
            continue

        builtin_name = v1_algo.pop(f'builtin{algo_type.title()}Name', None)
        if builtin_name is not None:
            v2_algo = {'name': builtin_name}

        else:
            code_directory = v1_algo.pop('codeDir')
            class_file_name = v1_algo.pop('classFileName')
            assert class_file_name.endswith('.py')
            class_name = class_file_name[:-3] + '.' + v1_algo.pop('className')
            v2_algo = {'className': class_name, 'codeDirectory': code_directory}

        if 'classArgs' in v1_algo:
            v2_algo['classArgs'] = v1_algo.pop('classArgs')

        v2[algo_type] = v2_algo
        _deprecate(v1_algo, v2, 'includeIntermediateResults')
        _move_field(v1_algo, v2, 'gpuIndices', 'tunerGpuIndices')

        if v1_algo:
            _logger.error('%s config not fully converted: %s', algo_type, v1_algo)

    ts = {'platform': platform}
    v2['trainingService'] = ts

    if platform == 'local':
        local_config = v1.pop('localConfig', {})
        _move_field(local_config, ts, 'gpuIndices')
        _move_field(local_config, ts, 'maxTrialNumPerGpu', 'maxTrialNumberPerGpu')
        _move_field(local_config, ts, 'useActiveGpu')
        if local_config:
            _logger.error('localConfig not fully converted: %s', local_config)

    if platform == 'remote':
        remote_config = v1.pop('remoteConfig', {})
        _move_field(remote_config, ts, 'reuse', 'reuseMode')
        if remote_config:
            _logger.error('remoteConfig not fully converted: %s', remote_config)

        ts['machineList'] = []
        for v1_machine in v1.pop('machineList'):
            v2_machine = {}
            ts['machineList'].append(v2_machine)
            _move_field(v1_machine, v2_machine, 'ip', 'host')
            _move_field(v1_machine, v2_machine, 'port')
            _move_field(v1_machine, v2_machine, 'username', 'user')
            _move_field(v1_machine, v2_machine, 'sshKeyPath', 'sshKeyFile')
            _move_field(v1_machine, v2_machine, 'passphrase')
            _move_field(v1_machine, v2_machine, 'gpuIndices')
            _move_field(v1_machine, v2_machine, 'maxTrialNumPerGpu', 'maxTrialNumberPerGpu')
            _move_field(v1_machine, v2_machine, 'useActiveGpu')
            _move_field(v1_machine, v2_machine, 'pythonPath')
            _move_field(v1_machine, v2_machine, 'passwd', 'password')
            if v1_machine:
                _logger.error('remote machine not fully converted: %s', v1_machine)

    if platform == 'openpai':
        _move_field(v1_trial, ts, 'nniManagerNFSMountPath', 'localStorageMountPoint')
        _move_field(v1_trial, ts, 'containerNFSMountPath', 'containerStorageMountPoint')
        _move_field(v1_trial, ts, 'cpuNum', 'trialCpuNumber')
        _move_field(v1_trial, ts, 'memoryMB', 'trialMemorySize')
        _move_field(v1_trial, ts, 'image', 'dockerImage')
        _move_field(v1_trial, ts, 'virtualCluster')
        _move_field(v1_trial, ts, 'paiStorageConfigName', 'storageConfigName')
        _move_field(v1_trial, ts, 'paiConfigPath', 'openpaiConfigFile')

        pai_config = v1.pop('paiConfig')
        _move_field(pai_config, ts, 'userName', 'username')
        _deprecate(pai_config, v2, 'password')
        _move_field(pai_config, ts, 'token')
        _move_field(pai_config, ts, 'host')
        _move_field(pai_config, ts, 'reuse', 'reuseMode')
        _move_field(pai_config, ts, 'gpuNum', 'trialGpuNumber')
        _move_field(pai_config, ts, 'cpuNum', 'trialCpuNumber')
        _move_field(pai_config, ts, 'memoryMB', 'trialMemorySize')
        _deprecate(pai_config, v2, 'maxTrialNumPerGpu')
        _deprecate(pai_config, v2, 'useActiveGpu')
        if pai_config:
            _logger.error('paiConfig not fully converted: %s', pai_config)

    if platform == 'aml':
        _move_field(v1_trial, ts, 'image', 'dockerImage')

        aml_config = v1.pop('amlConfig', {})
        _move_field(aml_config, ts, 'subscriptionId')
        _move_field(aml_config, ts, 'resourceGroup')
        _move_field(aml_config, ts, 'workspaceName')
        _move_field(aml_config, ts, 'computeTarget')
        _move_field(aml_config, ts, 'maxTrialNumPerGpu', 'maxTrialNumberPerGpu')
        _deprecate(aml_config, v2, 'useActiveGpu')
        if aml_config:
            _logger.error('amlConfig not fully converted: %s', aml_config)

    if platform == 'kubeflow':
        kf_config = v1.pop('kubeflowConfig')
        _move_field(kf_config, ts, 'operator')
        _move_field(kf_config, ts, 'apiVersion')

        storage_name = kf_config.pop('storage', None)
        if storage_name is None:
            storage_name = 'nfs' if 'nfs' in kf_config else 'azureStorage'
        if storage_name == 'nfs':
            nfs = kf_config.pop('nfs')
            ts['storage'] = {'storageType': 'nfs', 'server': nfs['server'], 'path': nfs['path']}
        if storage_name == 'azureStorage':
            key_vault = kf_config.pop('keyVault')
            azure_storage = kf_config.pop('azureStorage')
            ts['storage'] = {
                'storageType': 'azureStorage',
                'azureAccount': azure_storage['accountName'],
                'azureShare': azure_storage['azureShare'],
                'keyVaultName': key_vault['vaultName'],
                'keyVaultKey': key_vault['name'],
            }
            _deprecate(kf_config, v2, 'uploadRetryCount')

        if kf_config:
            _logger.error('kubeflowConfig not fully converted: %s', kf_config)

        _drop_field(v1_trial, 'nasMode')
        for role_name in ['worker', 'ps', 'master']:
            if role_name not in v1_trial:
                continue
            v1_role = v1_trial.pop(role_name)
            v2_role = {}
            ts[role_name] = v2_role

            _move_field(v1_role, v2_role, 'replicas')
            _move_field(v1_role, v2_role, 'command')
            _move_field(v1_role, v2_role, 'gpuNum', 'gpuNumber')
            _move_field(v1_role, v2_role, 'cpuNum', 'cpuNumber')
            _move_field(v1_role, v2_role, 'memoryMB', 'memorySize')
            _move_field(v1_role, v2_role, 'image', 'dockerImage')
            _deprecate(v1_role, v2, 'privateRegistryAuthPath')

            v2_role['codeDirectory'] = v2['trialCodeDirectory']

            if v1_role:
                _logger.error('kubeflow role not fully converted: %s', v1_role)

    if platform == 'frameworkcontroller':
        fc_config = v1.pop('frameworkcontrollerConfig')
        _move_field(fc_config, ts, 'serviceAccountName')
        _move_field(fc_config, ts, 'reuse', 'reuseMode')

        storage_name = fc_config.pop('storage', None)
        if storage_name is None:
            storage_name = 'nfs' if 'nfs' in fc_config else 'azureStorage'
        if storage_name == 'nfs':
            nfs = fc_config.pop('nfs')
            ts['storage'] = {'storageType': 'nfs', 'server': nfs['server'], 'path': nfs['path']}
        if storage_name == 'azureStorage':
            key_vault = fc_config.pop('keyVault')
            azure_storage = fc_config.pop('azureStorage')
            ts['storage'] = {
                'storageType': 'azureStorage',
                'azureAccount': azure_storage['accountName'],
                'azureShare': azure_storage['azureShare'],
                'keyVaultName': key_vault['vaultName'],
                'keyVaultKey': key_vault['name'],
            }
            _deprecate(fc_config, v2, 'uploadRetryCount')

        if fc_config:
            _logger.error('frameworkcontroller not fully converted: %s', fc_config)

        _drop_field(v1_trial, 'nasMode')
        ts['taskRoles'] = []
        for v1_role in v1_trial.pop('taskRoles', []):
            v2_role = {}
            ts['taskRoles'].append(v2_role)

            _move_field(v1_role, v2_role, 'name')
            _move_field(v1_role, v2_role, 'taskNum', 'taskNumber')
            _move_field(v1_role, v2_role, 'frameworkControllerCompletionPolicy', 'frameworkAttemptCompletionPolicy')
            _move_field(v1_role, v2_role, 'command')
            _move_field(v1_role, v2_role, 'gpuNum', 'gpuNumber')
            _move_field(v1_role, v2_role, 'cpuNum', 'cpuNumber')
            _move_field(v1_role, v2_role, 'memoryMB', 'memorySize')
            _move_field(v1_role, v2_role, 'image', 'dockerImage')
            _deprecate(v1_role, v2, 'privateRegistryAuthPath')

            policy = 'frameworkAttemptCompletionPolicy'
            if v1_role[policy]:
                v2_role[policy] = {}
                _move_field(v1_role[policy], v2_role[policy], 'minFailedTaskCount')
                _move_field(v1_role[policy], v2_role[policy], 'minSucceededTaskCount', 'minSucceedTaskCount')
            if not v1_role[policy]:
                v1_role.pop(policy)

            if v1_role:
                _logger.error('frameworkcontroller role not fully converted: %s', v1_role)

            # this is required, seems a bug in nni manager
            if not v2.get('trialCommand'):
                v2['trialCommand'] = v2_role['command']

    # hybrid mode should always use v2 schema, so no need to handle here

    v1_storage = v1.pop('sharedStorage', None)
    if v1_storage:
        v2_storage = {}
        v2['sharedStorage'] = v2_storage

        _move_field(v1_storage, v2_storage, 'storageType')
        _move_field(v1_storage, v2_storage, 'nfsServer')
        _move_field(v1_storage, v2_storage, 'exportedDirectory')
        _move_field(v1_storage, v2_storage, 'localMountPoint')
        _move_field(v1_storage, v2_storage, 'remoteMountPoint')
        _move_field(v1_storage, v2_storage, 'localMounted')
        _move_field(v1_storage, v2_storage, 'storageAccountName')
        _move_field(v1_storage, v2_storage, 'storageAccountKey')
        _move_field(v1_storage, v2_storage, 'containerName')

        if v1_storage:
            _logger.error('shared storage not fully converted: %s', v1_storage)

    if v1_trial:
        _logger.error('trial config not fully converted: %s', v1_trial)
    if v1:
        _logger.error('Config not fully converted: %s', v1)
    return v2

def _move_field(v1, v2, v1_key, v2_key=None):
    if v2_key is None:
        v2_key = v1_key
    if v1_key in v1:
        value = v1.pop(v1_key, None)
        if value is not None:
            v2[v2_key] = value

def _drop_field(v1, key):
    if key in v1:
        _logger.warning(f'Config field "{key}" is no longer supported and has been ignored')
        v1.pop(key)

def _deprecate(v1, v2, key):
    _drop_field(v1, key)

def convert_algo(algo_type, v1_algo):
    builtin_name = v1_algo.pop(f'builtin{algo_type.title()}Name', None)
    if builtin_name is not None:
        v2_algo = {'name': builtin_name}

    else:
        code_directory = v1_algo.pop('codeDir')
        class_file_name = v1_algo.pop('classFileName')
        assert class_file_name.endswith('.py')
        class_name = class_file_name[:-3] + '.' + v1_algo.pop('className')
        v2_algo = {'className': class_name, 'codeDirectory': code_directory}

    if 'classArgs' in v1_algo:
        v2_algo['classArgs'] = v1_algo.pop('classArgs')

    return v2_algo
