# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging

from .common import ExperimentConfig, AlgorithmConfig, CustomAlgorithmConfig
from .remote import RemoteMachineConfig
from .kubeflow import KubeflowRoleConfig, KubeflowNfsConfig, KubeflowAzureStorageConfig
from .frameworkcontroller import FrameworkControllerRoleConfig
from .shared_storage import NfsConfig, AzureBlobConfig
from . import util

_logger = logging.getLogger(__name__)

def to_v2(v1) -> ExperimentConfig:
    v1 = copy.deepcopy(v1)
    platform = v1.pop('trainingServicePlatform')
    assert platform in ['local', 'remote', 'openpai', 'aml']
    v2 = ExperimentConfig(platform)

    _drop_field(v1, 'authorName')
    _move_field(v1, v2, 'experimentName', 'experiment_name')
    _drop_field(v1, 'description')
    _move_field(v1, v2, 'trialConcurrency', 'trial_concurrency')
    _move_field(v1, v2, 'maxExecDuration', 'max_experiment_duration')
    if isinstance(v2.max_experiment_duration, (int, float)):
        v2.max_experiment_duration = str(v2.max_experiment_duration) + 's'
    _move_field(v1, v2, 'maxTrialNum', 'max_trial_number')
    _move_field(v1, v2, 'searchSpacePath', 'search_space_file')
    assert not v1.pop('multiPhase', None), 'Multi-phase is no longer supported'
    _deprecate(v1, v2, 'multiThread')
    _move_field(v1, v2, 'nniManagerIp', 'nni_manager_ip')
    _move_field(v1, v2, 'logDir', 'experiment_working_directory')
    _move_field(v1, v2, 'debug', 'debug')
    _deprecate(v1, v2, 'versionCheck')
    _move_field(v1, v2, 'logLevel', 'log_level')
    _deprecate(v1, v2, 'logCollection')
    v1.pop('useAnnotation', None)  # TODO: how to handle annotation in nni.Experiment?

    if 'trial' in v1:
        v1_trial = v1.pop('trial')
        _move_field(v1_trial, v2, 'command', 'trial_command')
        _move_field(v1_trial, v2, 'codeDir', 'trial_code_directory')
        _move_field(v1_trial, v2, 'gpuNum', 'trial_gpu_number')

    for algo_type in ['tuner', 'assessor', 'advisor']:
        if algo_type in v1:
            convert_algo(algo_type, v1, v2)

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
            ts.machine_list.append(v2_machine)
            _move_field(v1_machine, v2_machine, 'ip', 'host')
            _move_field(v1_machine, v2_machine, 'port', 'port')
            _move_field(v1_machine, v2_machine, 'username', 'user')
            _move_field(v1_machine, v2_machine, 'sshKeyPath', 'ssh_key_file')
            _move_field(v1_machine, v2_machine, 'passphrase', 'ssh_passphrase')
            _move_field(v1_machine, v2_machine, 'gpuIndices', 'gpu_indices')
            _move_field(v1_machine, v2_machine, 'maxTrialNumPerGpu', 'max_trial_number_per_gpu')
            _move_field(v1_machine, v2_machine, 'useActiveGpu', 'use_active_gpu')
            _move_field(v1_machine, v2_machine, 'pythonPath', 'python_path')
            _move_field(v1_machine, v2_machine, 'passwd', 'password')
            assert not v1_machine, v1_machine

    if platform == 'openpai':
        _move_field(v1_trial, ts, 'nniManagerNFSMountPath', 'local_storage_mount_point')
        _move_field(v1_trial, ts, 'containerNFSMountPath', 'container_storage_mount_point')
        _move_field(v1_trial, ts, 'cpuNum', 'trial_cpu_number')
        if 'memoryMB' in v1_trial:
            ts.trial_memory_size = str(v1_trial.pop('memoryMB')) + 'mb'
        _move_field(v1_trial, ts, 'image', 'docker_image')
        _deprecate(v1_trial, v2, 'virtualCluster')
        _move_field(v1_trial, ts, 'paiStorageConfigName', 'storage_config_name')
        _move_field(v1_trial, ts, 'paiConfigPath', 'openpaiConfigFile')

        pai_config = v1.pop('paiConfig')
        _move_field(pai_config, ts, 'userName', 'username')
        _deprecate(pai_config, v2, 'password')
        _move_field(pai_config, ts, 'token', 'token')
        _move_field(pai_config, ts, 'host', 'host')
        _move_field(pai_config, ts, 'reuse', 'reuse_mode')
        _move_field(pai_config, ts, 'gpuNum', 'trial_gpu_number')
        _move_field(pai_config, ts, 'cpuNum', 'trial_cpu_number')
        if 'memoryMB' in pai_config:
            ts.trial_memory_size = str(pai_config.pop('memoryMB')) + 'mb'
        _deprecate(pai_config, v2, 'maxTrialNumPerGpu')
        _deprecate(pai_config, v2, 'useActiveGpu')
        assert not pai_config, pai_config

    if platform == 'aml':
        _move_field(v1_trial, ts, 'image', 'docker_image')

        aml_config = v1.pop('amlConfig', {})
        _move_field(aml_config, ts, 'subscriptionId', 'subscription_id')
        _move_field(aml_config, ts, 'resourceGroup', 'resource_group')
        _move_field(aml_config, ts, 'workspaceName', 'workspace_name')
        _move_field(aml_config, ts, 'computeTarget', 'compute_target')
        _move_field(aml_config, ts, 'maxTrialNumPerGpu', 'max_trial_number_per_gpu')
        _deprecate(aml_config, v2, 'useActiveGpu')
        assert not aml_config, aml_config

    if platform == 'kubeflow':
        kf_config = v1.pop('kubeflowConfig')
        _move_field(kf_config, ts, 'operator', 'operator')
        ps_name = 'ps' if ts.operator != 'pytorch-operator' else 'master'
        _move_field(kf_config, ts, 'apiVersion', 'api_version')

        # FIXME: use storage service
        storage_name = kf_config.pop('storage', None)
        if storage_name is None:
            storage_name = 'nfs' if 'nfs' in kf_config else 'azureStorage'
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
            _deprecate(kf_config, v2, 'uploadRetryCount')

        assert not kf_config, kf_config

        _drop_field(v1_trial, 'nasMode')
        for role_name in [ps_name, 'worker']:
            if role_name not in v1_trial:
                continue
            v1_role = v1_trial.pop(role_name)
            v2_role = KubeflowRoleConfig()
            if role_name == 'worker':
                ts.worker = v2_role
            else:
                ts.parameter_server = v2_role

            _move_field(v1_role, v2_role, 'replicas', 'replicas')
            _move_field(v1_role, v2_role, 'command', 'command')
            _move_field(v1_role, v2_role, 'gpu_num', 'gpu_number')
            _move_field(v1_role, v2_role, 'cpu_num', 'cpu_number')
            v2_role.memory_size = str(v1_role.pop('memoryMB')) + 'mb'
            _move_field(v1_role, v2_role, 'image', 'docker_image')
            _deprecate(v1_role, v2, 'privateRegistryAuthPath')
            assert not v1_role, v1_role

    if platform == 'frameworkcontroller':
        fc_config = v1.pop('frameworkcontroller')
        _deprecate(fc_config, v2, 'serviceAccountName')

        storage_name = fc_config.pop('storage', None)
        if storage_name is None:
            storage_name = 'nfs' if 'nfs' in fc_config else 'azureStorage'
        if storage_name == 'nfs':
            nfs = fc_config.pop('nfs')
            ts.storage = KubeflowNfsConfig(server=nfs['server'], path=nfs['path'])
        if storage_name == 'azureStorage':
            key_vault = fc_config.pop('keyVault')
            azure_storage = fc_config.pop('azureStorage')
            ts.storage = KubeflowAzureStorageConfig(
                azure_account=azure_storage['accountName'],
                azure_share=azure_storage['azureShare'],
                key_vault=key_vault['vaultName'],
                key_vault_secret=key_vault['name']
            )
            _deprecate(fc_config, v2, 'uploadRetryCount')

        assert not fc_config, fc_config

        _drop_field(v1_trial, 'nasMode')
        ts.task_roles = []
        for v1_role in v1_trial.pop('taskRoles', []):
            v2_role = FrameworkControllerRoleConfig()
            ts.task_roles.append(v2_role)

            _move_field(v1_role, v2_role, 'name', 'name')
            _move_field(v1_role, v2_role, 'taskNum', 'task_number')
            policy = v1_role.pop('frameworkControllerCompletionPolicy', {})
            _move_field(policy, v2_role, 'minFailedTaskCount', 'attempt_completion_min_failed_tasks')
            _move_field(policy, v2_role, 'minSucceededTaskCount', 'attempt_completion_min_succeeded_tasks')
            _move_field(v1_role, v2_role, 'command', 'command')
            _move_field(v1_role, v2_role, 'gpuNum', 'gpu_number')
            _move_field(v1_role, v2_role, 'cpuNum', 'cpu_number')
            v2_role.memory_size = str(v1_role.pop('memoryMB')) + 'mb'
            _move_field(v1_role, v2_role, 'image', 'docker_image')
            _deprecate(v1_role, v2, 'privateRegistryAuthPath')
            assert not v1_role, v1_role

    # hybrid mode should always use v2 schema, so no need to handle here

    v1_storage = v1.pop('sharedStorage', None)
    if v1_storage:
        type_ = v1_storage.pop('storageType')
        if type_ == 'NFS':
            v2.shared_storage = NfsConfig(**v1_storage)
        elif type_ == 'AzureBlob':
            v2.shared_storage = AzureBlobConfig(**v1_storage)
        else:
            raise ValueError(f'bad storage type: {type_}')

    assert not v1_trial, v1_trial
    assert not v1, v1
    return v2.canonical()

def _move_field(v1, v2, v1_key, v2_key):
    if v1_key in v1:
        value = v1.pop(v1_key, None)
        if value is not None:
            setattr(v2, v2_key, value)

def _drop_field(v1, key):
    if key in v1:
        logging.warning(f'Configuration field {key} is no longer supported and has been ignored')
        v1.pop(key)

# NOTE: fields not yet supported by v2 are also (temporarily) placed here
def _deprecate(v1, v2, key):
    if key in v1:
        if v2._deprecated is None:
            v2._deprecated = {}
        v2._deprecated[key] = v1.pop(key)

def convert_algo(algo_type, v1, v2):
    if algo_type not in v1:
        return None
    v1_algo = v1.pop(algo_type)

    builtin_name = v1_algo.pop(f'builtin{algo_type.title()}Name', None)
    class_args = v1_algo.pop('classArgs', None)

    if builtin_name is not None:
        v2_algo = AlgorithmConfig(name=builtin_name, class_args=class_args)

    else:
        code_directory = util.canonical_path(v1_algo.pop('codeDir'))
        class_file_name = v1_algo.pop('classFileName')
        assert class_file_name.endswith('.py')
        class_name = class_file_name[:-3] + '.' + v1_algo.pop('className')
        v2_algo = CustomAlgorithmConfig(
            class_name=class_name,
            code_directory=code_directory,
            class_args=class_args
        )

    setattr(v2, algo_type, v2_algo)
    _deprecate(v1_algo, v2, 'includeIntermediateResults')
    _move_field(v1_algo, v2, 'gpuIndices', 'tuner_gpu_indices')
    assert not v1_algo, v1_algo
    return v2_algo
