from copy import deepcopy
import os.path

from nni.experiment.config import ExperimentConfig, LocalConfig

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
    'experimentWorkingDirectory': os.path.expanduser('~/nni-experiments'),
    'tuner': {'name': 'random'},
    'trainingService': {
        'platform': 'local',
        'trialCommand': 'python main.py',
        'trialCodeDirectory': os.path.realpath('.'),
        'maxTrialNumberPerGpu': 1,
        'reuseMode': False,
    },
}

## detailed config ##

detailed_json = {
    'experimentName': 'test case',
    'searchSpaceFile': 'assets/search_space.json',
    'trialCommand': 'python main.py',
    'trialCodeDirectory': 'assets',
    'trialConcurrency': 2,
    'trialGpuNumber': 1,
    'maxExperimentDuration': '1.5h',
    'maxTrialNumber': 10,
    'maxTrialDuration': 60,
    'nniManagerIp': '1.2.3.4',
    'debug': True,
    'logLevel': 'warning',
    'tunerGpuIndices': 0,
    'assessor': {
        'name': 'assess',
    },
    'advisor': {
        'className': 'Advisor',
        'codeDirectory': 'assets',
        'classArgs': {'random_seed': 0},
    },
    'trainingService': {
        'platform': 'local',
        'useActiveGpu': False,
        'maxTrialNumberPerGpu': 2,
        'gpuIndices': '1,2',
        'reuseMode': True,
    },
    'sharedStorage': {
        'storageType': 'NFS',
        'localMountPoint': 'assets',
        'remoteMountPoint': '/tmp',
        'localMounted': 'nomount',
        'nfsServer': 'nfs.test.case',
        'exportedDirectory': 'root',
    },
}

detailed_canon = deepcopy(detailed_json)
detailed_canon['searchSpaceFile'] = os.path.realpath('assets/search_space.json')
detailed_canon['searchSpace'] = {'a': 1}
detailed_canon['trialCodeDirectory'] = os.path.realpath('assets')
detailed_canon['maxExperimentDuration'] = 1.5 * 3600
detailed_canon['useAnnotation'] = False
detailed_canon['experimentWorkingDirectory'] = os.path.expanduser('~/nni-experiments')
detailed_canon['tunerGpuIndices'] = [0]
detailed_canon['advisor']['codeDirectory'] = os.path.realpath('assets')
detailed_canon['trainingService']['trialCommand'] = 'python main.py'
detailed_canon['trainingService']['trialCodeDirectory'] = os.path.realpath('assets')
detailed_canon['trainingService']['trialGpuNumber'] = 1
detailed_canon['trainingService']['gpuIndices'] = [1, 2]
detailed_canon['sharedStorage']['localMountPoint'] = os.path.realpath('assets')

## test function ##

def test_all():
    minimal = ExperimentConfig(**minimal_json)
    assert minimal.json() == minimal_canon

    assert minimal_class.json() == minimal_canon

    detailed = ExperimentConfig(**detailed_json)
    assert detailed.json() == detailed_canon

if __name__ == '__main__':
    test_all()
