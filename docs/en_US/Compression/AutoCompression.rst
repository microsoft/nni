Auto Compression with NNI Experiment
====================================

If you want to compress your model, but don't know what compression algorithm to choose, or don't know what sparsity is suitable for your model, or just want to try more possibilities, auto compression may help you.
Users can choose different compression algorithms and define the algorithms' search space, then auto compression will launch an NNI experiment and try different compression algorithms with varying sparsity automatically. 
Of course, in addition to the sparsity rate, users can also introduce other related parameters into the search space.
If you don't know what is search space or how to write search space, `this <./Tutorial/SearchSpaceSpec.rst>`__ is for your reference.
Auto compression using experience is similar to the NNI experiment in python.
The main differences are as follows:

* Use a generator to help generate search space object.
* Need to provide the model to be compressed, and the model should have already been pre-trained.
* No need to set ``trial_command``, additional need to set ``auto_compress_module`` as ``AutoCompressionExperiment`` input.

Generate search space
---------------------

Due to the extensive use of nested search space, we recommend a using generator to configure search space.
The following is an example. Using ``add_config()`` add subconfig, then ``dumps()`` search space dict.

.. code-block:: python

    from nni.algorithms.compression.pytorch.auto_compress import AutoCompressionSearchSpaceGenerator

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

Users need to inherit ``AbstractAutoCompressionModule`` and override the abstract class function.

.. code-block:: python

    from nni.algorithms.compression.pytorch.auto_compress import AbstractAutoCompressionModule

    class AutoCompressionModule(AbstractAutoCompressionModule):
        @classmethod
        def model(cls) -> nn.Module:
            ...
            return _model

        @classmethod
        def evaluator(cls) -> Callable[[nn.Module], float]:
            ...
            return _evaluator

Users need to implement at least ``model()`` and ``evaluator()``.
If you use iterative pruner, you need to additional implement ``optimizer_factory()``, ``criterion()`` and ``sparsifying_trainer()``.
If you want to finetune the model after compression, you need to implement ``optimizer_factory()``, ``criterion()``, ``post_compress_finetuning_trainer()`` and ``post_compress_finetuning_epochs()``.
The ``optimizer_factory()`` should return a factory function, the input is an iterable variable, i.e. your ``model.parameters()``, and the output is an optimizer instance.
The two kinds of ``trainer()`` should return a trainer with input ``model, optimizer, criterion, current_epoch``.
The full abstract interface refers to :githublink:`interface.py <nni/algorithms/compression/pytorch/auto_compress/interface.py>`.
An example of ``AutoCompressionModule`` implementation refers to :githublink:`auto_compress_module.py <examples/model_compress/auto_compress/torch/auto_compress_module.py>`.

Launch NNI experiment
---------------------

Similar to launch from python, the difference is no need to set ``trial_command`` and put the user-provided ``AutoCompressionModule`` as ``AutoCompressionExperiment`` input.

.. code-block:: python

    from pathlib import Path
    from nni.algorithms.compression.pytorch.auto_compress import AutoCompressionExperiment

    from auto_compress_module import AutoCompressionModule

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
