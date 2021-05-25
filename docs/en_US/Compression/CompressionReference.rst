Model Compression API Reference
===============================

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


Module Wrapper
^^^^^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.compressor.PrunerModuleWrapper
    :members:


..  autoclass:: nni.compression.pytorch.compressor.QuantizerModuleWrapper
    :members:

Weight Masker
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

Model Speedup
-------------

Quantization Speedup
^^^^^^^^^^^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.quantization_speedup.backend.BaseModelSpeedup
    :members:

..  autoclass:: nni.compression.pytorch.quantization_speedup.integrated_tensorrt.ModelSpeedupTensorRT
    :members:

..  autoclass:: nni.compression.pytorch.quantization_speedup.calibrator.Calibrator
    :members:


Compression Utilities
---------------------

Sensitivity Utilities
^^^^^^^^^^^^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.utils.sensitivity_analysis.SensitivityAnalysis
    :members:

Topology Utilities
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

Model FLOPs/Parameters Counter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  autofunction:: nni.compression.pytorch.utils.counter.count_flops_params
