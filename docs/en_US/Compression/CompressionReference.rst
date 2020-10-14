
Python API Reference of Compression Utilities
=============================================

.. contents::

Sensitivity Utilities
---------------------

..  autoclass:: nni.compression.torch.utils.sensitivity_analysis.SensitivityAnalysis
    :members:

Topology Utilities
------------------

..  autoclass:: nni.compression.torch.utils.shape_dependency.ChannelDependency
    :members:
..  autoclass:: nni.compression.torch.utils.shape_dependency.GroupDependency
    :members:
..  autoclass:: nni.compression.torch.utils.mask_conflict.CatMaskPadding
    :members:
..  autoclass:: nni.compression.torch.utils.mask_conflict.GroupMaskConflict
    :members:
..  autoclass:: nni.compression.torch.utils.mask_conflict.ChannelMaskConflict
    :members:
Model FLOPs/Parameters Counter
------------------------------

..  autofunction:: nni.compression.torch.utils.counter.count_flops_params
