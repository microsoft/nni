使用 NNI Experiment 自动压缩
================================================

如果你想压缩你的模型，但不知道该选择什么压缩算法，或者不知道什么稀疏度适合你的模型，或者只是想尝试更多的可能性，自动压缩可能会帮助你。
用户可以选择不同的压缩算法，并定义算法的搜索空间，然后自动压缩将启动一个 NNI 实验，并自动尝试不同稀疏度的压缩算法。 
当然，除了稀疏度之外，用户还可以在搜索空间中引入其他相关参数。
如果你不知道什么是搜索空间或如何编写搜索空间，可以参考 `此教程 <./Tutorial/SearchSpaceSpec.rst>`__ 。
在 Python 中使用自动压缩与 NNI Experiment 很相似。
主要区别如下：

* 使用生成器帮助生成搜索空间对象
* 需要提供要压缩的模型，并且模型应该已经过预训练
* 不需要设置 ``trial_command``，需要额外设置 ``auto_compress_module`` 作为 ``AutoCompressionExperiment`` 的输入。

生成搜索空间
---------------------

由于大量使用嵌套搜索空间，我们建议使用生成器来配置搜索空间。
示例如下： 使用 ``add_config()`` 增加子配置，然后 ``dumps()`` 搜索空间字典。

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

目前我们支持以下 Pruner 和 Quantizer：

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
