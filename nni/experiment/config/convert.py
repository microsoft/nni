import json
from tempfile import NamedTemporaryFile
from typing import Any, Dict

from .common import ExperimentConfig
from . import util

def to_old_yaml(config: ExperimentConfig) -> Dict[str, Any]:
    config.validate()
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
        'command': ' && '.join(data.pop('trialCommand')),
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
                'preCommand': ' && '.join(machine['trialPrepareCommand'])
            }
            prepare_command = machine['trialPrepareCommand']
            if prepare_command is not None:
                machine['preCommand'] = ' && '.join(prepare_command)

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
