# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Example showing how to create experiment with Python code.
"""

from pathlib import Path

from nni.experiment import Experiment, RemoteMachineConfig

search_space = {
    "dropout_rate": { "_type": "uniform", "_value": [0.5, 0.9] },
    "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
    "hidden_size": { "_type": "choice", "_value": [124, 512, 1024] },
    "batch_size": { "_type": "choice", "_value": [16, 32] },
    "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
}

experiment = Experiment(['local', 'remote'])
experiment.config.experiment_name = 'test'
experiment.config.trial_concurrency = 3
experiment.config.max_trial_number = 10
experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 mnist.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service[0].use_active_gpu = True
experiment.config.training_service[1].reuse_mode = True
rm_conf = RemoteMachineConfig()
rm_conf.host = '10.1.1.1'
rm_conf.user = 'xxx'
rm_conf.password = 'xxx'
rm_conf.port = 22
experiment.config.training_service[1].machine_list = [rm_conf]

experiment.run(26780, debug=True)
