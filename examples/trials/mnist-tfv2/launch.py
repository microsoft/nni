# FIXME: For demonstration only. It should not be here

from pathlib import Path

from nni.experiment import Experiment
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

tuner = HyperoptTuner('tpe')

search_space = {
    "dropout_rate": { "_type": "uniform", "_value": [0.5, 0.9] },
    "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
    "hidden_size": { "_type": "choice", "_value": [124, 512, 1024] },
    "batch_size": { "_type": "choice", "_value": [16, 32] },
    "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
}

experiment = Experiment(tuner, 'local')
experiment.config.experiment_name = 'test'
experiment.config.trial_concurrency = 2
experiment.config.max_trial_number = 5
experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 mnist.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.training_service.use_active_gpu = True

experiment.run(8081)
