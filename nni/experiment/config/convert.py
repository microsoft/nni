# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

from .common import ExperimentConfig, AlgorithmConfig, CustomAlgorithmConfig
from .remote import RemoteConfig, RemoteMachineConfig
from . import util

_logger = logging.getLogger(__name__)

def to_v2(v1) -> ExperimentConfig:
    platform = v1.pop('trainingServicePlatform')
    assert platform in ['local', 'remote', 'openpai']
    v2 = ExperimentConfig(platform)

    _drop_field(v1, 'authorName')
    _move_field(v1, v2, 'experimentName', 'experiment_name')
    _drop_field(v1, 'description')
    _move_field(v1, v2, 'trialConcurrency', 'trial_concurrency')
    _move_field(v1, v2, 'maxExecDuration', 'max_experiment_duration')
    _move_field(v1, v2, 'maxTrialNum', 'max_trial_number')
    _move_field(v1, v2, 'searchSpacePath', 'search_space_file')
    _drop_field(v1, 'multiPhase')
    _drop_field(v1, 'multiThread')
    _move_field(v1, v2, 'nniManagerIp', 'nni_manager_ip')
    _move_field(v1, v2, 'logDir', 'experiment_working_directory')
    _move_field(v1, v2, 'debug', 'debug')
    _drop_field(v1, 'versionCheck')
    _move_field(v1, v2, 'logLevel', 'log_level')
    _drop_field(v1, 'logCollection')
    _drop_field(v1, 'useAnnotation')

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

        else:
            class_directory = util.canonical_path(v1_algo.pop('codeDir'))
            class_file_name = v1_algo.pop('classFileName')
            assert class_file_name.endswith('.py')
            class_name = class_file_name[:-3] + '.' + v1_algo.pop('className')
            v2_algo = CustomAlgorithmConfig(
                class_name=class_name,
                class_directory=class_directory,
                class_args=class_args
            )

        setattr(v2, algo_type, v2_algo)
        _drop_field(v1_algo, 'includeIntermediateResults')
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
        _move_field(v1_trial, ts, 'memoryMB', 'trial_memory_size')  # FIXME: unit
        _move_field(v1_trial, ts, 'image', 'docker_image')
        _drop_field(v1_trial, 'virtualCluster')  # FIXME: better error message
        _move_field(v1_trial, ts, 'paiStorageConfigName', 'storage_config_name')
        _move_field(v1_trial, ts, 'paiConfigPath', 'openpaiConfigFile')

        pai_config = v1.pop('paiConfig')
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

    assert not v1_trial, v1_trial
    assert not v1, v1
    return v2.canonical()

def _drop_field(v1, key):
    if key in v1:
        logging.warning(f'Configuration field {key} is no longer supported and has been ignored')
        v1.pop(key)

def _move_field(v1, v2, v1_key, v2_key):
    if v1_key in v1:
        value = v1.pop(v1_key, None)
        if value is not None:
            setattr(v2, v2_key, value)


def to_v1_yaml(config: ExperimentConfig, skip_nnictl: bool = False) -> Dict[str, Any]:
    config.validate(False)
    data = config.json()

    ts = data.pop('trainingService')

    data['trial'] = {
        'command': data.pop('trialCommand'),
        'codeDir': data.pop('trialCodeDirectory'),
    }

    if 'trialGpuNumber' in data:
        data['trial']['gpuNum'] = data.pop('trialGpuNumber')

    if isinstance(ts, list):
        hybrid_names = []
        for conf in ts:
            if conf['platform'] == 'openpai':
                conf['platform'] = 'pai'
            hybrid_names.append(conf['platform'])
            _handle_training_service(conf, data)
        data['trainingServicePlatform'] = 'hybrid'
        data['hybridConfig'] = {'trainingServicePlatforms': hybrid_names}
    else:
        if ts['platform'] == 'openpai':
            ts['platform'] = 'pai'
        data['trainingServicePlatform'] = ts['platform']
        _handle_training_service(ts, data)

    data['authorName'] = 'N/A'
    data['experimentName'] = data.get('experimentName', 'N/A')
    data['maxExecDuration'] = data.pop('maxExperimentDuration', '999d')
    if data['debug']:
        data['versionCheck'] = False
    data['maxTrialNum'] = data.pop('maxTrialNumber', 99999)

    ss = data.pop('searchSpace', None)
    ss_file = data.pop('searchSpaceFile', None)
    if ss is not None:
        ss_file = NamedTemporaryFile('w', delete=False)
        json.dump(ss, ss_file, indent=4)
        data['searchSpacePath'] = ss_file.name
    elif ss_file is not None:
        data['searchSpacePath'] = ss_file
    if 'experimentWorkingDirectory' in data:
        data['logDir'] = data.pop('experimentWorkingDirectory')

    for algo_type in ['tuner', 'assessor', 'advisor']:
        algo = data.get(algo_type)
        if algo is None:
            continue
        if algo['name'] is not None:  # builtin
            algo['builtin' + algo_type.title() + 'Name'] = algo.pop('name')
            algo.pop('className', None)
            algo.pop('codeDirectory', None)
        else:
            algo.pop('name', None)
            class_name_parts = algo.pop('className').split('.')
            algo['codeDir'] = algo.pop('codeDirectory', '') + '/'.join(class_name_parts[:-2])
            algo['classFileName'] = class_name_parts[-2] + '.py'
            algo['className'] = class_name_parts[-1]

    tuner_gpu_indices = _convert_gpu_indices(data.pop('tunerGpuIndices', None))
    if tuner_gpu_indices is not None:
        data['tuner']['gpuIndicies'] = tuner_gpu_indices

    return data

def _handle_training_service(ts, data):
    if ts['platform'] == 'local':
        data['localConfig'] = {
            'useActiveGpu': ts.get('useActiveGpu', False),
            'maxTrialNumPerGpu': ts['maxTrialNumberPerGpu']
        }
        if 'gpuIndices' in ts:
            data['localConfig']['gpuIndices'] = _convert_gpu_indices(ts['gpuIndices'])

    elif ts['platform'] == 'remote':
        data['remoteConfig'] = {'reuse': ts['reuseMode']}
        data['machineList'] = []
        for machine in ts['machineList']:
            machine_v1 = {
                'ip': machine.get('host'),
                'port': machine.get('port'),
                'username': machine.get('user'),
                'passwd': machine.get('password'),
                'sshKeyPath': machine.get('sshKeyFile'),
                'passphrase': machine.get('sshPassphrase'),
                'gpuIndices': _convert_gpu_indices(machine.get('gpuIndices')),
                'maxTrialNumPerGpu': machine.get('maxTrialNumPerGpu'),
                'useActiveGpu': machine.get('useActiveGpu'),
                'pythonPath': machine.get('pythonPath')
            }
            machine_v1 = {k: v for k, v in machine_v1.items() if v is not None}
            data['machineList'].append(machine_v1)

    elif ts['platform'] == 'pai':
        data['trial']['image'] = ts['dockerImage']
        data['trial']['nniManagerNFSMountPath'] = ts['localStorageMountPoint']
        data['trial']['containerNFSMountPath'] = ts['containerStorageMountPoint']
        data['trial']['paiStorageConfigName'] = ts['storageConfigName']
        data['trial']['cpuNum'] = ts['trialCpuNumber']
        data['trial']['memoryMB'] = ts['trialMemorySize']
        data['paiConfig'] = {
            'userName': ts['username'],
            'token': ts['token'],
            'host': ts['host'],
            'reuse': ts['reuseMode']
        }
        if 'openpaiConfigFile' in ts:
            data['paiConfig']['paiConfigPath'] = ts['openpaiConfigFile']
        elif 'openpaiConfig' in ts:
            conf_file = NamedTemporaryFile('w', delete=False)
            json.dump(ts['openpaiConfig'], conf_file, indent=4)
            data['paiConfig']['paiConfigPath'] = conf_file.name

    elif ts['platform'] == 'aml':
        data['trial']['image'] = ts['dockerImage']
        data['amlConfig'] = dict(ts)
        data['amlConfig'].pop('platform')
        data['amlConfig'].pop('dockerImage')

    elif ts['platform'] == 'kubeflow':
        data['trial'].pop('command')
        data['trial'].pop('gpuNum')
        data['kubeflowConfig'] = dict(ts['storage'])
        data['kubeflowConfig']['operator'] = ts['operator']
        data['kubeflowConfig']['apiVersion'] = ts['apiVersion']
        data['trial']['worker'] = _convert_kubeflow_role(ts['worker'])
        if ts.get('parameterServer') is not None:
            if ts['operator'] == 'tf-operator':
                data['trial']['ps'] = _convert_kubeflow_role(ts['parameterServer'])
            else:
                data['trial']['master'] = _convert_kubeflow_role(ts['parameterServer'])

    elif ts['platform'] == 'frameworkcontroller':
        data['trial'].pop('command')
        data['trial'].pop('gpuNum')
        data['frameworkcontrollerConfig'] = dict(ts['storage'])
        data['frameworkcontrollerConfig']['serviceAccountName'] = ts['serviceAccountName']
        data['trial']['taskRoles'] = [_convert_fxctl_role(r) for r in ts['taskRoles']]

    elif ts['platform'] == 'adl':
        data['trial']['image'] = ts['dockerImage']

def _convert_gpu_indices(indices):
    return ','.join(str(idx) for idx in indices) if indices is not None else None

def _convert_kubeflow_role(data):
    return {
        'replicas': data['replicas'],
        'command': data['command'],
        'gpuNum': data['gpuNumber'],
        'cpuNum': data['cpuNumber'],
        'memoryMB': util.parse_size(data['memorySize']),
        'image': data['dockerImage']
    }

def _convert_fxctl_role(data):
    return {
        'name': data['name'],
        'taskNum': data['taskNumber'],
        'command': data['command'],
        'gpuNum': data['gpuNumber'],
        'cpuNum': data['cpuNumber'],
        'memoryMB': util.parse_size(data['memorySize']),
        'image': data['dockerImage'],
        'frameworkAttemptCompletionPolicy': {
            'minFailedTaskCount': data['attemptCompletionMinFailedTasks'],
            'minSucceededTaskCount': data['attemptCompletionMinSucceededTasks']
        }
    }


def to_cluster_metadata(config: ExperimentConfig) -> List[Dict[str, Any]]:
    experiment_config = to_v1_yaml(config, skip_nnictl=True)
    ret = []

    if isinstance(config.training_service, list):
        hybrid_conf = dict()
        hybrid_conf['hybrid_config'] = experiment_config['hybridConfig']
        for conf in config.training_service:
            metadata = _get_cluster_metadata(conf.platform, experiment_config)
            if metadata is not None:
                hybrid_conf.update(metadata)
        ret.append(hybrid_conf)
    else:
        metadata = _get_cluster_metadata(config.training_service.platform, experiment_config)
        if metadata is not None:
            ret.append(metadata)

    if experiment_config.get('nniManagerIp') is not None:
        ret.append({'nni_manager_ip': {'nniManagerIp': experiment_config['nniManagerIp']}})
    ret.append({'trial_config': experiment_config['trial']})
    return ret

def _get_cluster_metadata(platform: str, experiment_config) -> Dict:
    if platform == 'local':
        request_data = dict()
        request_data['local_config'] = experiment_config['localConfig']
        if request_data['local_config']:
            if request_data['local_config'].get('gpuIndices') and isinstance(request_data['local_config'].get('gpuIndices'), int):
                request_data['local_config']['gpuIndices'] = str(request_data['local_config'].get('gpuIndices'))
        return request_data

    elif platform == 'remote':
        request_data = dict()
        if experiment_config.get('remoteConfig'):
            request_data['remote_config'] = experiment_config['remoteConfig']
        else:
            request_data['remote_config'] = {'reuse': False}
        request_data['machine_list'] = experiment_config['machineList']
        if request_data['machine_list']:
            for i in range(len(request_data['machine_list'])):
                if isinstance(request_data['machine_list'][i].get('gpuIndices'), int):
                    request_data['machine_list'][i]['gpuIndices'] = str(request_data['machine_list'][i].get('gpuIndices'))
        return request_data

    elif platform == 'openpai':
        return {'pai_config': experiment_config['paiConfig']}

    elif platform == 'aml':
        return {'aml_config': experiment_config['amlConfig']}

    elif platform == 'kubeflow':
        return {'kubeflow_config': experiment_config['kubeflowConfig']}

    elif platform == 'frameworkcontroller':
        return {'frameworkcontroller_config': experiment_config['frameworkcontrollerConfig']}

    elif platform == 'adl':
        return None

    else:
        raise RuntimeError('Unsupported training service ' + platform)

def to_rest_json(config: ExperimentConfig) -> Dict[str, Any]:
    experiment_config = to_v1_yaml(config, skip_nnictl=True)
    request_data = dict()
    request_data['authorName'] = experiment_config['authorName']
    request_data['experimentName'] = experiment_config['experimentName']
    request_data['trialConcurrency'] = experiment_config['trialConcurrency']
    request_data['maxExecDuration'] = util.parse_time(experiment_config['maxExecDuration'])
    request_data['maxTrialNum'] = experiment_config['maxTrialNum']

    if config.search_space is not None:
        request_data['searchSpace'] = json.dumps(config.search_space)
    elif config.search_space_file is not None:
        request_data['searchSpace'] = Path(config.search_space_file).read_text()

    request_data['trainingServicePlatform'] = experiment_config.get('trainingServicePlatform')
    if experiment_config.get('advisor'):
        request_data['advisor'] = experiment_config['advisor']
        if request_data['advisor'].get('gpuNum'):
            _logger.warning('gpuNum is deprecated, please use gpuIndices instead.')
        if request_data['advisor'].get('gpuIndices') and isinstance(request_data['advisor'].get('gpuIndices'), int):
            request_data['advisor']['gpuIndices'] = str(request_data['advisor'].get('gpuIndices'))
    elif experiment_config.get('tuner'):
        request_data['tuner'] = experiment_config['tuner']
        if request_data['tuner'].get('gpuNum'):
            _logger.warning('gpuNum is deprecated, please use gpuIndices instead.')
        if request_data['tuner'].get('gpuIndices') and isinstance(request_data['tuner'].get('gpuIndices'), int):
            request_data['tuner']['gpuIndices'] = str(request_data['tuner'].get('gpuIndices'))
        if 'assessor' in experiment_config:
            request_data['assessor'] = experiment_config['assessor']
            if request_data['assessor'].get('gpuNum'):
                _logger.warning('gpuNum is deprecated, please remove it from your config file.')
    else:
        request_data['tuner'] = {'builtinTunerName': '_user_created_'}
    #debug mode should disable version check
    if experiment_config.get('debug') is not None:
        request_data['versionCheck'] = not experiment_config.get('debug')
    #validate version check
    if experiment_config.get('versionCheck') is not None:
        request_data['versionCheck'] = experiment_config.get('versionCheck')
    if experiment_config.get('logCollection'):
        request_data['logCollection'] = experiment_config.get('logCollection')
    request_data['clusterMetaData'] = []
    if experiment_config['trainingServicePlatform'] == 'local':
        if experiment_config.get('localConfig'):
            request_data['clusterMetaData'].append(
                {'key': 'local_config', 'value': experiment_config['localConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'remote':
        request_data['clusterMetaData'].append(
            {'key': 'machine_list', 'value': experiment_config['machineList']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
        if not experiment_config.get('remoteConfig'):
            # set default value of reuse in remoteConfig to False
            experiment_config['remoteConfig'] = {'reuse': False}
        request_data['clusterMetaData'].append(
            {'key': 'remote_config', 'value': experiment_config['remoteConfig']})
    elif experiment_config['trainingServicePlatform'] == 'pai':
        request_data['clusterMetaData'].append(
            {'key': 'pai_config', 'value': experiment_config['paiConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'kubeflow':
        request_data['clusterMetaData'].append(
            {'key': 'kubeflow_config', 'value': experiment_config['kubeflowConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'frameworkcontroller':
        request_data['clusterMetaData'].append(
            {'key': 'frameworkcontroller_config', 'value': experiment_config['frameworkcontrollerConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    elif experiment_config['trainingServicePlatform'] == 'aml':
        request_data['clusterMetaData'].append(
            {'key': 'aml_config', 'value': experiment_config['amlConfig']})
        request_data['clusterMetaData'].append(
            {'key': 'trial_config', 'value': experiment_config['trial']})
    return request_data
