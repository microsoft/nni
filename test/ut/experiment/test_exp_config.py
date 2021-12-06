import os.path
from pathlib import Path

from nni.experiment.config import ExperimentConfig

def expand_path(path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), path))

## minimal config ##

minimal_json = {
    'searchSpace': {'a': 1},
    'trialCommand': 'python main.py',
    'trialConcurrency': 2,
    'tuner': {
        'name': 'random',
    },
    'trainingService': {
        'platform': 'local',
    },
}

minimal_class = ExperimentConfig('local')
minimal_class.search_space = {'a': 1}
minimal_class.trial_command = 'python main.py'
minimal_class.trial_concurrency = 2
minimal_class.tuner.name = 'random'

minimal_canon = {
    'searchSpace': {'a': 1},
    'trialCommand': 'python main.py',
    'trialCodeDirectory': os.path.realpath('.'),
    'trialConcurrency': 2,
    'useAnnotation': False,
    'debug': False,
    'logLevel': 'info',
    'experimentWorkingDirectory': str(Path.home() / 'nni-experiments'),
    'tuner': {'name': 'random'},
    'trainingService': {
        'platform': 'local',
        'trialCommand': 'python main.py',
        'trialCodeDirectory': os.path.realpath('.'),
        'debug': False,
        'maxTrialNumberPerGpu': 1,
        'reuseMode': False,
    },
}

## detailed config ##

detailed_canon = {
    'experimentName': 'test case',
    'searchSpaceFile': expand_path('assets/search_space.json'),
    'searchSpace': {'a': 1},
    'trialCommand': 'python main.py',
    'trialCodeDirectory': expand_path('assets'),
    'trialConcurrency': 2,
    'trialGpuNumber': 1,
    'maxExperimentDuration': '1.5h',
    'maxTrialNumber': 10,
    'maxTrialDuration': 60,
    'nniManagerIp': '1.2.3.4',
    'useAnnotation': False,
    'debug': True,
    'logLevel': 'warning',
    'experimentWorkingDirectory': str(Path.home() / 'nni-experiments'),
    'tunerGpuIndices': [0],
    'assessor': {
        'name': 'assess',
    },
    'advisor': {
        'className': 'Advisor',
        'codeDirectory': expand_path('assets'),
        'classArgs': {'random_seed': 0},
    },
    'trainingService': {
        'platform': 'local',
        'trialCommand': 'python main.py',
        'trialCodeDirectory': expand_path('assets'),
        'trialGpuNumber': 1,
        'debug': True,
        'useActiveGpu': False,
        'maxTrialNumberPerGpu': 2,
        'gpuIndices': [1, 2],
        'reuseMode': True,
    },
    'sharedStorage': {
        'storageType': 'NFS',
        'localMountPoint': expand_path('assets'),
        'remoteMountPoint': '/tmp',
        'localMounted': 'usermount',
        'nfsServer': 'nfs.test.case',
        'exportedDirectory': 'root',
    },
}

## test function ##

def test_all():
    minimal = ExperimentConfig(**minimal_json)
    assert minimal.json() == minimal_canon

    assert minimal_class.json() == minimal_canon

    detailed = ExperimentConfig.load(expand_path('assets/config.yaml'))
    assert detailed.json() == detailed_canon

if __name__ == '__main__':
    test_all()
