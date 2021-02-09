Compression Reference
=====================

.. contents::

Compressors
-----------

Compressor
^^^^^^^^^^

..  autoclass:: nni.compression.pytorch.compressor.Compressor
    :members:


..  autoclass:: nni.compression.pytorch.compressor.Pruner
    :members:

..  autoclass:: nni.algorithms.compression.pytorch.pruning.one_shot.OneshotPruner
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

..  autoclass:: nni.algorithms.compression.pytorch.pruning.structured_pruning.StructuredWeightMasker
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
