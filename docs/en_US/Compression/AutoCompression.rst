Auto Compression with NNI Experiment
====================================

This approach is mainly a combination of compression and nni experiments.
It allows users to define compressor search space, including types, parameters, etc.
Its using experience is similar to launch the NNI experiment from python.
The main differences are as follows:

* Use a generator to help generate search space object.
* Need to provide the model to be compressed, and the model should have already pre-trained.
* No need to set ``trial_command``, additional need to set ``auto_compress_module_file_name``.

Generate search space
---------------------

Due to the extensive use of nested search space, we recommend using generator to configure search space.
The following is an example. Using ``add_pruner_config()`` and ``add_quantizer_config()`` add subconfig, then ``dumps()`` search space dict.

.. code-block:: python

    from nni.algorithms.compression.pytorch.auto_compress import AutoCompressSearchSpaceGenerator

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

Now we support the following pruners and quantizers:

.. code-block:: python

    PRUNER_DICT = {
        'level': LevelPruner,
        'slim': SlimPruner,
        'l1': L1FilterPruner,
        'l2': L2FilterPruner,
        'fpgm': FPGMPruner,
        'taylorfo': TaylorFOWeightFilterPruner,
        'apoz': ActivationAPoZRankFilterPruner,
        'mean_activation': ActivationMeanRankFilterPruner
    }

    QUANTIZER_DICT = {
        'naive': NaiveQuantizer,
        'qat': QAT_Quantizer,
        'dorefa': DoReFaQuantizer,
        'bnn': BNNQuantizer
    }

Provide user model for compression
----------------------------------

Users need to inherit ``AbstractAutoCompressModule`` and override the abstract class function.

.. code-block:: python

    from nni.algorithms.compression.pytorch.auto_compress import AbstractAutoCompressModule

    class AutoCompressModule(AbstractAutoCompressModule):
        @classmethod
        def model(cls) -> nn.Module:
            ...
            return _model

        @classmethod
        def evaluator(cls) -> Callable[[nn.Module], float]:
            ...
            return _evaluator

Users need to implement at least ``model()`` and ``evaluator()``.
If you use iterative pruner, you need to additional implement ``optimizer()``, ``criterion()`` and ``sparsifying_trainer()``.
If you want to finetune the model after compression, you need to implement ``post_compress_finetuning_trainer()``.
The path of file that contains the ``AutoCompressModule`` needs to be specified in experiment config.
The full abstract interface refers to :githublink:`interface.py <nni/algorithms/compression/pytorch/auto_compress/interface.py>`.
An example of ``AutoCompressModule`` implementation refers to :githublink:`auto_compress_module.py <examples/model_compress/auto_compress/torch/auto_compress_module.py>`.

Launch NNI experiment
---------------------

Similar to launch from python, the difference is no need to set ``trial_command``.
By default, ``auto_compress_module_file_name`` is set as ``./auto_compress_module.py``.
Remember that ``auto_compress_module_file_name`` is the relative file path under ``trial_code_directory``.

.. code-block:: python

    from pathlib import Path
    from nni.algorithms.compression.pytorch.auto_compress import AutoCompressExperiment

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
