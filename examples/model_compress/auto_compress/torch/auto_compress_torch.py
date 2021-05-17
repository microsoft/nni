# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

from nni.algorithms.compression.pytorch.auto_compress import AutoCompressExperiment, AutoCompressSearchSpaceGenerator

generator = AutoCompressSearchSpaceGenerator()
generator.add_pruner_config('level', [
    {
        "sparsity": {
            "_type": "uniform",
            "_value": [0.01, 0.99]
        },
        'op_types': ['default']
    }
])
generator.add_pruner_config('l1', [
    {
        "sparsity": {
            "_type": "uniform",
            "_value": [0.01, 0.99]
        },
        'op_types': ['Conv2d']
    }
])
generator.add_quantizer_config('qat', [
    {
        'quant_types': ['weight', 'output'],
        'quant_bits': {
            'weight': 8,
            'output': 8
        },
        'op_types': ['Conv2d', 'Linear']
    }])
search_space = generator.dumps()

experiment = AutoCompressExperiment('local')
experiment.config.experiment_name = 'auto compress torch example'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 10
experiment.config.search_space = search_space
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

# the relative file path under trial_code_directory, which contains the class AutoCompressModule
experiment.config.auto_compress_module_file_name = './auto_compress_module.py'

experiment.run(8088)
