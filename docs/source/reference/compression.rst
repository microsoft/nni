Compression API Reference
=========================

Pruner
------

Please refer to :doc:`../compression/pruner`.

Quantizer
---------

Please refer to :doc:`../compression/quantizer`.

Pruning Speedup
---------------

.. autoclass:: nni.compression.pytorch.speedup.ModelSpeedup
    :members:

Quantization Speedup
--------------------

.. autoclass:: nni.compression.pytorch.quantization_speedup.ModelSpeedupTensorRT
    :members:

Compression Utilities
---------------------

.. autoclass:: nni.compression.pytorch.utils.sensitivity_analysis.SensitivityAnalysis
    :members:

.. autoclass:: nni.compression.pytorch.utils.shape_dependency.ChannelDependency
    :members:

.. autoclass:: nni.compression.pytorch.utils.shape_dependency.GroupDependency
    :members:

.. autoclass:: nni.compression.pytorch.utils.mask_conflict.ChannelMaskConflict
    :members:

.. autoclass:: nni.compression.pytorch.utils.mask_conflict.GroupMaskConflict
    :members:

.. autofunction:: nni.compression.pytorch.utils.counter.count_flops_params

.. autofunction:: nni.algorithms.compression.v2.pytorch.utils.pruning.compute_sparsity

Framework Related
-----------------

.. autoclass:: nni.algorithms.compression.v2.pytorch.base.Pruner
    :members:

.. autoclass:: nni.algorithms.compression.v2.pytorch.base.PrunerModuleWrapper

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.basic_pruner.BasicPruner
    :members:

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.tools.DataCollector
    :members:

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.tools.MetricsCalculator
    :members:

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.tools.SparsityAllocator
    :members:

.. autoclass:: nni.algorithms.compression.v2.pytorch.base.BasePruningScheduler
    :members:

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.tools.TaskGenerator
    :members:

.. autoclass:: nni.compression.pytorch.compressor.Quantizer
    :members:

.. autoclass:: nni.compression.pytorch.compressor.QuantizerModuleWrapper
    :members:

.. autoclass:: nni.compression.pytorch.compressor.QuantGrad
    :members:
