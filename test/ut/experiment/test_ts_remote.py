import os.path
from pathlib import Path

from nni.experiment.config import ExperimentConfig, AlgorithmConfig, RemoteConfig, RemoteMachineConfig

## minimal config ##

minimal_json = {
    'searchSpace': {'a': 1},
    'trialCommand': 'python main.py',
    'trialConcurrency': 2,
    'tuner': {
        'name': 'random',
    },
    'trainingService': {
        'platform': 'remote',
        'machine_list': [
            {
                'host': '1.2.3.4',
                'user': 'test_user',
                'password': '123456',
            },
        ],
    },
}

minimal_class = ExperimentConfig(
    search_space = {'a': 1},
    trial_command = 'python main.py',
    trial_concurrency = 2,
    tuner = AlgorithmConfig(
        name = 'random',
    ),
    training_service = RemoteConfig(
        machine_list = [
            RemoteMachineConfig(
                host = '1.2.3.4',
                user = 'test_user',
                password = '123456',
            ),
        ],
    ),
)

minimal_canon = {
    'experimentType': 'hpo',
    'searchSpace': {'a': 1},
    'trialCommand': 'python main.py',
    'trialCodeDirectory': os.path.realpath('.'),
    'trialConcurrency': 2,
    'useAnnotation': False,
    'debug': False,
    'logLevel': 'info',
    'experimentWorkingDirectory': str(Path.home() / 'nni-experiments'),
    'tuner': {
        'name': 'random',
    },
    'trainingService': {
        'platform': 'remote',
        'trialCommand': 'python main.py',
        'trialCodeDirectory': os.path.realpath('.'),
        'debug': False,
        'machineList': [
            {
                'host': '1.2.3.4',
                'port': 22,
                'user': 'test_user',
                'password': '123456',
                'useActiveGpu': False,
                'maxTrialNumberPerGpu': 1,
            }
        ],
        'reuseMode': False,
        #'logCollection': 'on_error',
    }
}

## detailed config ##

detailed_json = {
    'searchSpace': {'a': 1},
    'trialCommand': 'python main.py',
    'trialConcurrency': 2,
    'trialGpuNumber': 1,
    'nni_manager_ip': '1.2.3.0',
    'tuner': {
        'name': 'random',
    },
    'trainingService': {
        'platform': 'remote',
        'machine_list': [
            {
                'host': '1.2.3.4',
                'user': 'test_user',
                'password': '123456',
            },
            {
                'host': '1.2.3.5',
                'user': 'test_user_2',
                'password': 'abcdef',
                'use_active_gpu': True,
                'max_trial_number_per_gpu': 2,
                'gpu_indices': '0,1',
                'python_path': '~/path',  # don't do this in actual experiment
            },
        ],
    },
}

detailed_canon = {
    'experimentType': 'hpo',
    'searchSpace': {'a': 1},
    'trialCommand': 'python main.py',
    'trialCodeDirectory': os.path.realpath('.'),
    'trialConcurrency': 2,
    'trialGpuNumber': 1,
    'nniManagerIp': '1.2.3.0',
    'useAnnotation': False,
    'debug': False,
    'logLevel': 'info',
    'experimentWorkingDirectory': str(Path.home() / 'nni-experiments'),
    'tuner': {'name': 'random'},
    'trainingService': {
        'platform': 'remote',
        'trialCommand': 'python main.py',
        'trialCodeDirectory': os.path.realpath('.'),
        'trialGpuNumber': 1,
        'nniManagerIp': '1.2.3.0',
        'debug': False,
        'machineList': [
            {
                'host': '1.2.3.4',
                'port': 22,
                'user': 'test_user',
                'password': '123456',
                'useActiveGpu': False,
                'maxTrialNumberPerGpu': 1
            },
            {
                'host': '1.2.3.5',
                'port': 22,
                'user': 'test_user_2',
                'password': 'abcdef',
                'useActiveGpu': True,
                'maxTrialNumberPerGpu': 2,
                'gpuIndices': [0, 1],
                'pythonPath': '~/path'
            }
        ],
        'reuseMode': False,
        #'logCollection': 'on_error',
    }
}

## test function ##

def test_remote():
    config = ExperimentConfig(**minimal_json)
    assert config.json() == minimal_canon

    assert minimal_class.json() == minimal_canon

    config = ExperimentConfig(**detailed_json)
    assert config.json() == detailed_canon

if __name__ == '__main__':
    test_remote()
