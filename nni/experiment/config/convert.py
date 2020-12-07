# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

from .common import ExperimentConfig
from . import util

_logger = logging.getLogger(__name__)


def to_old_yaml(config: ExperimentConfig, skip_nnictl: bool = False) -> Dict[str, Any]:
    config.validate(skip_nnictl)
    data = config.json()

    ts = data.pop('trainingService')
    if ts['platform'] == 'openpai':
        ts['platform'] = 'pai'

    data['authorName'] = 'N/A'
    data['experimentName'] = data.get('experimentName', 'N/A')
    data['maxExecDuration'] = data.pop('maxExperimentDuration', '999d')
    if data['debug']:
        data['versionCheck'] = False
    data['maxTrialNum'] = data.pop('maxTrialNumber', 99999)
    data['trainingServicePlatform'] = ts['platform']
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

    data['trial'] = {
        'command': data.pop('trialCommand'),
        'codeDir': data.pop('trialCodeDirectory'),
        'gpuNum': data.pop('trialGpuNumber', '')
    }

    if ts['platform'] == 'local':
        data['localConfig'] = {
            'useActiveGpu': ts['useActiveGpu'],
            'maxTrialNumPerGpu': ts['maxTrialNumberPerGpu']
        }
        if ts.get('gpuIndices') is not None:
            data['localConfig']['gpuIndices'] = ','.join(str(idx) for idx in ts['gpuIndices'])

    elif ts['platform'] == 'remote':
        data['remoteConfig'] = {'reuse': ts['reuseMode']}
        data['machineList'] = []
        for machine in ts['machineList']:
            machine = {
                'ip': machine['host'],
                'username': machine['user'],
                'passwd': machine['password'],
                'sshKeyPath': machine['sshKeyFile'],
                'passphrase': machine['sshPassphrase'],
                'gpuIndices': _convert_gpu_indices(machine['gpuIndices']),
                'maxTrialNumPerGpu': machine['maxTrialNumPerGpu'],
                'useActiveGpu': machine['useActiveGpu'],
                'preCommand': machine['trialPrepareCommand']
            }

    elif ts['platform'] == 'pai':
        data['trial']['cpuNum'] = ts['trialCpuNumber']
        data['trial']['memoryMB'] = util.parse_size(ts['trialMemorySize'])
        data['trial']['image'] = ts['docker_image']
        data['paiConfig'] = {
            'userName': ts['username'],
            'token': ts['token'],
            'host': 'https://' + ts['host'],
            'reuse': ts['reuseMode']
        }

    return data

def _convert_gpu_indices(indices):
    return ','.join(str(idx) for idx in indices) if indices is not None else None


def to_cluster_metadata(config: ExperimentConfig) -> List[Dict[str, Any]]:
    experiment_config = to_old_yaml(config, skip_nnictl=True)
    ret = []

    if config.training_service.platform == 'local':
        request_data = dict()
        request_data['local_config'] = experiment_config['localConfig']
        if request_data['local_config']:
            if request_data['local_config'].get('gpuIndices') and isinstance(request_data['local_config'].get('gpuIndices'), int):
                request_data['local_config']['gpuIndices'] = str(request_data['local_config'].get('gpuIndices'))
            if request_data['local_config'].get('maxTrialNumOnEachGpu'):
                request_data['local_config']['maxTrialNumOnEachGpu'] = request_data['local_config'].get('maxTrialNumOnEachGpu')
            if request_data['local_config'].get('useActiveGpu'):
                request_data['local_config']['useActiveGpu'] = request_data['local_config'].get('useActiveGpu')
        ret.append(request_data)

    elif config.training_service.platform == 'remote':
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
        ret.append(request_data)

    elif config.training_service.platform == 'openpai':
        pai_config_data = dict()
        pai_config_data['pai_config'] = experiment_config['paiConfig']
        ret.append(pai_config_data)

    else:
        raise RuntimeError('Unsupported training service ' + config.training_service.platform)

    if experiment_config.get('nniManagerIp') is not None:
        ret.append({'nni_manager_ip': {'nniManagerIp': experiment_config['nniManagerIp']}})
    ret.append({'trial_config': experiment_config['trial']})
    return ret


def to_rest_json(config: ExperimentConfig) -> Dict[str, Any]:
    experiment_config = to_old_yaml(config, skip_nnictl=True)
    request_data = dict()
    request_data['authorName'] = experiment_config['authorName']
    request_data['experimentName'] = experiment_config['experimentName']
    request_data['trialConcurrency'] = experiment_config['trialConcurrency']
    request_data['maxExecDuration'] = util.parse_time(experiment_config['maxExecDuration'])
    request_data['maxTrialNum'] = experiment_config['maxTrialNum']

    if config.search_space is not None:
        request_data['searchSpace'] = json.dumps(config.search_space)
    else:
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
        request_data['clusterMetaData'].append(
            {'key':'codeDir', 'value':experiment_config['trial']['codeDir']})
        request_data['clusterMetaData'].append(
            {'key': 'command', 'value': experiment_config['trial']['command']})
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
