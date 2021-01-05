模型压缩 Python API 参考
=============================================

.. contents::

灵敏度工具
---------------------

..  autoclass:: nni.compression.pytorch.utils.sensitivity_analysis.SensitivityAnalysis
    :members:

拓扑结构工具
------------------

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
------------------------------

..  autofunction:: nni.compression.pytorch.utils.counter.count_flops_params
