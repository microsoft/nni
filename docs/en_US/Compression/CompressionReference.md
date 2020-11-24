# Python API Reference of Compression Utilities

```eval_rst
.. contents::
```

## Sensitivity Utilities

```eval_rst
..  autoclass:: nni.compression.pytorch.utils.sensitivity_analysis.SensitivityAnalysis
    :members:

```

## Topology Utilities

```eval_rst
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

```

## Model FLOPs/Parameters Counter

```eval_rst
..  autofunction:: nni.compression.pytorch.utils.counter.count_flops_params

```
