# 模型压缩 Python API 参考

```eval_rst
.. contents::
```

## 灵敏度工具

```eval_rst
..  autoclass:: nni.compression.torch.utils.sensitivity_analysis.SensitivityAnalysis
    :members:

```

## 拓扑结构工具

```eval_rst
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

```

## 模型 FLOPs 和参数计数器

```eval_rst
..  autofunction:: nni.compression.torch.utils.counter.count_flops_params

```