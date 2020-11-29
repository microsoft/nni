import json
from tempfile import NamedTemporaryFile

from .common import ExperimentConfig

def to_legacy_yaml(config: ExperimentConfig) -> None:
    data = config.json()
    ts = data.pop('trainingService')

    data['authorName'] = 'N/A'
    data['experimentName'] = data.get('experimentName', 'N/A')
    data['maxExecDuration'] = data.pop('maxExperimentDuration', '999d')
    if data['debug']:
        data['versionCheck'] = False
    data['maxTrialNum'] = data.pop('maxTrialNumber', 99999)
    data['trainingServicePlatform'] = ts
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
        indices = data.pop('tunerGpuIndices')
        if isinstance(indices, list):
            indices = ','.join(str(idx) for idx in indices)
        data['tuner']['gpuIndicies'] = indices

    data['trial'] = {}
    data['trial']['command'] = _convert_commands(data.pop('trialCommand'))
    data['trial']['codDir'] = data.pop('trialCodeDirectory')
    data['trial']['gpuNum'] = data.pop('trialGpuNumber')

    # platform config here

    if 'reuseMode' in data:
        tsConfig['reuse'] = data.pop('reuseMode')


def _convert_commands(commands: Union[str, List[str]]) -> str:
    return commands if isinstance(commands, str) else ' && '.join(commands)
