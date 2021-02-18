# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from . import (
    AlgorithmConfig,
    CustomAlgorithmConfig,
    ExperimentConfig,
    KubeflowAzureStorageConfig,
    KubeflowNfsConfig,
    KubeflowRoleConfig,
    RemoteMachineConfig,
)
from . import util

def convert_to_v2(v1) -> ExperimentConfig:
    platform = v1.pop('trainingServicePlatform')
    if platform == 'pai':
        platform = 'openpai'
    v2 = ExperimentConfig(platform)

    # see nni/tools/nnictl/config_schema.py for v1 field list

    _drop_field(v1, 'authorName')
    _move_field(v1, v2, 'experimentName', 'experiment_name')
    _drop_field(v1, 'description')
    _move_field(v1, v2, 'trialConcurrency', 'trial_concurrency')
    _move_field(v1, v2, 'maxExecDuration', 'max_experiment_duration')
    _move_field(v1, v2, 'maxTrialNum', 'max_trial_number')
    v2.search_space_file = util.canonical_path(v1.pop('searchSpacePath', None))
    _drop_field(v1, 'multiPhase')
    _drop_field(v1, 'multiThread')  # FIXME
    _move_field(v1, v2, 'nniManagerIp', 'nni_manager_ip')
    _move_field(v1, v2, 'logDir', 'experiment_working_directory')
    _move_field(v1, v2, 'debug', 'debug')
    _drop_field(v1, 'versionCheck')
    _move_field(v1, v2, 'logLevel', 'log_level')
    _drop_field(v1, 'logCollection')
    _drop_field(v1, 'useAnnotation')  # FIXME

    if 'trial' in v1:
        v1_trial = v1.pop('trial')
        _move_field(v1_trial, v2, 'command', 'trial_command')
        _move_field(v1_trial, v2, 'codeDir', 'trial_code_directory')
        _move_field(v1_trial, v2, 'gpuNum', 'trial_gpu_number')

    for algo_type in ['tuner', 'assessor', 'advisor']:
        if algo_type not in v1:
            continue
        v1_algo = v1.pop(algo_type)

        builtin_name = v1_algo.pop(f'builtin{algo_type.title()}Name', None)
        class_args = v1_algo.pop('classArgs', None)

        if builtin_name is not None:
            v2_algo = AlgorithmConfig(name=builtin_name, class_args=class_args)
            # FIXME: check built-in name available

        else:
            class_directory = util.canonical_path(v1_algo.pop('codeDir'))
            class_file_name = v1_algo.pop('classFileName')
            assert class_file_name.endswith('.py')
            class_name = class_file_name[:-3]
            v2_algo = CustomAlgorithmConfig(
                class_name=class_name,
                class_directory=class_directory,
                class_args=class_args
            )

        setattr(v2, algo_type, v2_algo)
        _drop_field(v1_algo, 'includeIntermediateResults')  # FIXME
        _move_field(v1_algo, v2, 'gpuIndices', 'tuner_gpu_indices')
        assert not v1_algo, v1_algo

    ts = v2.training_service

    if platform == 'local':
        local_config = v1.pop('localConfig', {})
        _move_field(local_config, ts, 'gpuIndices', 'gpu_indices')
        _move_field(local_config, ts, 'maxTrialNumPerGpu', 'max_trial_number_per_gpu')
        _move_field(local_config, ts, 'useActiveGpu', 'use_active_gpu')
        assert not local_config, local_config

    if platform == 'remote':
        remote_config = v1.pop('remoteConfig', {})
        _move_field(remote_config, ts, 'reuse', 'reuse_mode')
        assert not remote_config, remote_config

        ts.machine_list = []
        for v1_machine in v1.pop('machineList'):
            v2_machine = RemoteMachineConfig()
            _move_field(v1_machine, v2_machine, 'ip', 'host')
            _move_field(v1_machine, v2_machine, 'port', 'port')
            _move_field(v1_machine, v2_machine, 'username', 'user')
            v2_machine.ssh_key_file = util.canonical_path(v1_machine.pop('sshKeyPath', None))
            _move_field(v1_machine, v2_machine, 'passphrase', 'ssh_passphrase')
            _move_field(v1_machine, v2_machine, 'gpuIndices', 'gpu_indices')
            _move_field(v1_machine, v2_machine, 'maxTrialNumPerGpu', 'max_trial_number_per_gpu')
            _move_field(v1_machine, v2_machine, 'useActiveGpu', 'use_active_gpu')
            _move_field(v1_machine, v2_machine, 'preCommand', 'trial_prepare_command')
            _move_field(v1_machine, v2_machine, 'passwd', 'password')
            ts.machine_list.append(v2_machine)
            assert not v1_machine, v1_machine

    if platform == 'openpai':
        _move_field(v1_trial, ts, 'nniManagerNFSMountPath', 'local_storage_mount_point')
        _move_field(v1_trial, ts, 'containerNFSMountPath', 'container_storage_mount_point')
        _move_field(v1_trial, ts, 'cpuNum', 'trial_cpu_number')
        _move_field(v1_trial, ts, 'memoryMB', 'trial_memory_size')  # FIXME: unit
        _move_field(v1_trial, ts, 'image', 'docker_image')
        _drop_field(v1_trial, 'virtualCluster')  # FIXME: better error message
        _move_field(v1_trial, ts, 'paiStorageConfigName', 'storage_config_name')
        ts.openpai_config_file = util.canonical_path(v1_trial.pop('paiConfigPath', None))

        pai_config = v1.pop('paiConfig', {})
        _move_field(pai_config, ts, 'userName', 'username')
        _drop_field(pai_config, 'password')
        _move_field(pai_config, ts, 'token', 'token')
        _move_field(pai_config, ts, 'host', 'host')
        _move_field(pai_config, ts, 'reuse', 'reuse_mode')
        _move_field(pai_config, ts, 'gpuNum', 'trial_gpu_number')
        _move_field(pai_config, ts, 'cpuNum', 'trial_cpu_number')
        _move_field(pai_config, ts, 'memoryMB', 'trial_memory_size')  # FIXME: unit
        #_move_field(pai_config, ts, 'maxTrialNumPerGpu', 'max_trial_number_per_gpu')  # FIXME
        #_move_field(pai_config, ts, 'useActiveGpu', 'use_active_gpu')  # FIXME
        assert not pai_config, pai_config

    if platform == 'aml':
        _move_field(v1_trial, ts, 'image', 'docker_image')

        aml_config = v1.pop('amlConfig', {})
        _move_field(aml_config, ts, 'subscriptionId', 'subscription_id')
        _move_field(aml_config, ts, 'resourceGroup', 'resource_group')
        _move_field(aml_config, ts, 'workspaceName', 'workspace_name')
        _move_field(aml_config, ts, 'computeTarget', 'compute_target')
        #_move_field(aml_config, ts, 'maxTrialNumPerGpu', 'max_trial_number_per_gpu')  # FIXME
        #_move_field(aml_config, ts, 'useActiveGpu', 'use_active_gpu')  # FIXME
        assert not aml_config, aml_config

    if platform == 'kubeflow':
        kf_config = v1.pop('kubeflowConfig')

        ts.operator = kf_config.pop('operator')
        if ts.operator == 'pytorch-operator':
            ps_name = 'master'
        else:
            ps_name = 'ps'

        _move_field(kf_config, ts, 'apiVersion', 'api_version')

        storage_name = kf_config.pop('storage')
        if storage_name == 'nfs':
            nfs = kf_config.pop('nfs')
            ts.storage = KubeflowNfsConfig(server=nfs['server'], path=nfs['path'])
        if storage_name == 'azureStorage':
            key_vault = kf_config.pop('keyVault')
            azure_storage = kf_config.pop('azureStorage')
            ts.storage = KubeflowAzureStorageConfig(
                azure_account=azure_storage['accountName'],
                azure_share=azure_storage['azureShare'],
                key_vault=key_vault['vaultName'],
                key_vault_secret=key_vault['name']
            )
            _drop_field(kf_config, 'uploadRetryCount'),  # FIXME

        assert not kf_config, kf_config

        for role in [ps_name, 'worker']:
            v1_role = v1_trial.pop(role)
            v2_role = KubeflowRoleConfig()
            _move_field(v1_role, v2_role, 'replicas', 'replicas')
            _move_field(v1_role, v2_role, 'command', 'command')
            _move_field(v1_role, v2_role, 'gpu_num', 'gpu_number')
            _move_field(v1_role, v2_role, 'cpu_num', 'cpu_number')
            _move_field(v1_role, v2_role, 'memoryMB', 'memory_size')  # FIXME: unit
            _move_field(v1_role, v2_role, 'image', 'docker_image')
            _drop_field(v1_role, 'privateRegistryAuthPath')  # FIXME
            if role == 'worker':
                ts.worker = v2_role
            else:
                ts.parameter_server = v2_role

        _drop_field(v1_trial, 'nasMode')

    if platform == 'frameworkcontroller':
        pass  # FIXME

    if platform == 'adl':
        pass  # FIXME

    if platform == 'hybrid':
        pass  # FIXME

    assert not v1_trial, v1_trial
    assert not v1, v1
    return v2

def _drop_field(v1, key):
    if key in v1:
        logging.warning(f'Configuration field {key} is no longer supported and has been ignored')
        v1.pop(key)

def _move_field(v1, v2, v1_key, v2_key):
    if v1_key in v1:
        value = v1.pop(v1_key, None)
        if value is not None:
            setattr(v2, v2_key, value)
