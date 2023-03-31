import pytest

from nni.experiment.config import *
from nni.nas.experiment.config import *

def test_execution_engine_config():
    assert isinstance(ExecutionEngineConfig(name='ts'), TrainingServiceEngineConfig)
    assert isinstance(ExecutionEngineConfig(name='sequential'), SequentialEngineConfig)
    assert TrainingServiceEngineConfig().json() == dict(name='ts')
    assert ExecutionEngineConfig(name='sequential', continue_on_failure=True, max_duration=1).max_duration == 1
    assert ExecutionEngineConfig(name='cgo', max_concurrency_cgo=1, batch_waiting_time=1).batch_waiting_time == 1

    with pytest.raises(ValueError, match='Invalid ExecutionEngineConfig subclass'):
        ExecutionEngineConfig(name='wrong')
    assert isinstance(ExecutionEngineConfig(), ExecutionEngineConfig)


def test_model_format_config():
    assert isinstance(ModelFormatConfig(name='raw'), RawModelFormatConfig)
    assert ModelFormatConfig(name='graph', dummy_input=[1, 2, 3]).dummy_input == [1, 2, 3]


def test_experiment_config():
    config = NasExperimentConfig('ts', 'simplified', 'local')
    config.trial_concurrency = 1
    config_json = config.json()

    assert config_json['trialCommand'].endswith(' trial')
    assert config_json['executionEngine'] == {'name': 'ts'}
    assert config_json['trialConcurrency'] == 1
    assert config_json['trainingService']['platform'] == 'local'
    assert config_json['trainingService']['trialCommand'] == config_json['trialCommand']
    assert config_json['modelFormat'] == {'name': 'simplified'}

    assert ExperimentConfig(**config_json).canonical_copy() == config.canonical_copy()
