# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

from nni.compression.pytorch.auto_compress import AutoCompressionExperiment, AutoCompressionSearchSpaceGenerator

from auto_compress_module import AutoCompressionModule

generator = AutoCompressionSearchSpaceGenerator()
generator.add_config('level', [
    {
        "sparsity": {
            "_type": "uniform",
            "_value": [0.01, 0.99]
        },
        'op_types': ['default']
    }
])
generator.add_config('l1', [
    {
        "sparsity": {
            "_type": "uniform",
            "_value": [0.01, 0.99]
        },
        'op_types': ['Conv2d']
    }
])
generator.add_config('qat', [
    {
        'quant_types': ['weight', 'output'],
        'quant_bits': {
            'weight': 8,
            'output': 8
        },
        'op_types': ['Conv2d', 'Linear']
    }])
search_space = generator.dumps()

experiment = AutoCompressionExperiment(AutoCompressionModule, 'local')
experiment.config.experiment_name = 'auto compression torch example'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 10
experiment.config.search_space = search_space
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

experiment.run(8088)
