import json
from pathlib import Path

import yaml

from nni.experiment.config import ExperimentConfig, AlgorithmConfig, LocalConfig

## template ##

config = ExperimentConfig(
    search_space_file = '',
    trial_command = 'echo hello',
    trial_concurrency = 1,
    tuner = AlgorithmConfig(name='randomm'),
    training_service = LocalConfig()
)

space_correct = {
    'pool_type': {
        '_type': 'choice',
        '_value': ['max', 'min', 'avg']
    },
    '学习率': {
        '_type': 'loguniform',
        '_value': [1e-7, 0.1]
    }
}

# FIXME
# PyYAML 6.0 (YAML 1.1) does not support tab and scientific notation
# JSON does not support comment and extra comma
# So some combinations will fail to load
formats = [
    ('ss_tab.json', 'JSON (tabs + scientific notation)'),
    ('ss_comma.json', 'JSON with extra comma'),
    #('ss_tab_comma.json', 'JSON (tabs + scientific notation) with extra comma'),
    ('ss.yaml', 'YAML'),
    #('ss_yaml12.yaml', 'YAML 1.2 with scientific notation'),
]

def test_search_space():
    for space_file, description in formats:
        try:
            config.search_space_file = Path(__file__).parent / 'assets' / space_file
            space = config.json()['searchSpace']
            assert space == space_correct
        except Exception as e:
            print('Failed to load search space format: ' + description)
            raise e

if __name__ == '__main__':
    test_search_space()
