import json
from tempfile import NamedTemporaryFile
from typing import Any, Dict

from .common import ExperimentConfig
from . import util

def to_old_yaml(config: ExperimentConfig) -> Dict[str, Any]:
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
    if 'searchSpace' in data:
        ss_file = NamedTemporaryFile('w', delete=False)
        json.dump(ss_file, data.pop('searchSpace'))
        data['searchSpacePath'] = ss_file.name
    elif 'searchSpaceFile' in data:
        data['searchSpacePath'] = data.pop('searchSpaceFile')
    if 'experimentWorkingDirectory' in data:
        data['logDir'] = data.pop('experimentWorkingDirectory')

    #for algo in ['tuner', 'assessor', 'advisor']:
    #    FIXME: check what to do with algorithm metadata PR

    if 'tunerGpuIndices' in data:
        indices = _convert_gpu_indices(data.pop('tunerGpuIndices'))
        if indices is not None:
            data['tuner']['gpuIndicies'] = indices

    data['trial'] = {
        'command': ' && '.join(data.pop('trialCommand')),
        'codDir': data.pop('trialCodeDirectory'),
        'gpuNum': data.pop('trialGpuNumber')
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
