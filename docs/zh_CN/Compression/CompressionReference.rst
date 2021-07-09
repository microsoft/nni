模型压缩 API 参考
=============================================

.. contents::

Compressors
-----------

Compressor
^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.compressor.Compressor
    :members:

..  autoclass:: nni.compression.pytorch.compressor.Pruner
    :members:

..  autoclass:: nni.compression.pytorch.compressor.Quantizer
    :members:


Module 的包装
^^^^^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.compressor.PrunerModuleWrapper
    :members:


..  autoclass:: nni.compression.pytorch.compressor.QuantizerModuleWrapper
    :members:

权重掩码
^^^^^^^^^^^^^
..  autoclass:: nni.algorithms.compression.pytorch.pruning.weight_masker.WeightMasker
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.structured_pruning_masker.StructuredWeightMasker
    :members:


Pruners
^^^^^^^
..  autoclass:: nni.algorithms.compression.pytorch.pruning.sensitivity_pruner.SensitivityPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.one_shot_pruner.OneshotPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.one_shot_pruner.LevelPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.one_shot_pruner.L1FilterPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.one_shot_pruner.L2FilterPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.one_shot_pruner.FPGMPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.iterative_pruner.IterativePruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.iterative_pruner.SlimPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.iterative_pruner.TaylorFOWeightFilterPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.iterative_pruner.ActivationAPoZRankFilterPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.iterative_pruner.ActivationMeanRankFilterPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.iterative_pruner.AGPPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.iterative_pruner.ADMMPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.auto_compress_pruner.AutoCompressPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.net_adapt_pruner.NetAdaptPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.simulated_annealing_pruner.SimulatedAnnealingPruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.lottery_ticket.LotteryTicketPruner
    :members:


Quantizers
^^^^^^^^^^
..  autoclass:: nni.algorithms.compression.pytorch.quantization.quantizers.NaiveQuantizer
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.quantization.quantizers.QAT_Quantizer
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.quantization.quantizers.DoReFaQuantizer
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.quantization.quantizers.BNNQuantizer
    :members:

模型加速
-------------

量化模型加速
^^^^^^^^^^^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.quantization_speedup.backend.BaseModelSpeedup
    :members:

..  autoclass:: nni.compression.pytorch.quantization_speedup.integrated_tensorrt.ModelSpeedupTensorRT
    :members:

..  autoclass:: nni.compression.pytorch.quantization_speedup.calibrator.Calibrator
    :members:


压缩工具
---------------------

灵敏度工具
^^^^^^^^^^^^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.utils.sensitivity_analysis.SensitivityAnalysis
    :members:

拓扑结构工具
^^^^^^^^^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.utils.shape_dependency.ChannelDependency
    :members:

..  autoclass:: nni.compression.pytorch.utils.shape_dependency.GroupDependency
    :members:

..  autoclass:: nni.compression.pytorch.utils.mask_conflict.CatMaskPadding
    :members:

..  autoclass:: nni.compression.pytorch.utils.mask_conflict.GroupMaskConflict
    :members:

..  autoclass:: nni.compression.pytorch.utils.mask_conflict.ChannelMaskConflict
    :members:

模型 FLOPs 和参数计数器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  autofunction:: nni.compression.pytorch.utils.counter.count_flops_params