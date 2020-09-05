# NAS 参考

```eval_rst
.. contents::
```

## Mutable

```eval_rst
..  autoclass:: nni.nas.pytorch.mutables.Mutable
    :members:

..  autoclass:: nni.nas.pytorch.mutables.LayerChoice
    :members:

..  autoclass:: nni.nas.pytorch.mutables.InputChoice
    :members:

..  autoclass:: nni.nas.pytorch.mutables.MutableScope
    :members:
```

### 工具

```eval_rst
..  autofunction:: nni.nas.pytorch.utils.global_mutable_counting
```

## Mutator

```eval_rst
..  autoclass:: nni.nas.pytorch.base_mutator.BaseMutator
    :members:

..  autoclass:: nni.nas.pytorch.mutator.Mutator
    :members:
```

### Random Mutator

```eval_rst
..  autoclass:: nni.nas.pytorch.random.RandomMutator
    :members:
```

### 工具

```eval_rst
..  autoclass:: nni.nas.pytorch.utils.StructuredMutableTreeNode
    :members:
```

## Trainer

### Trainer

```eval_rst
..  autoclass:: nni.nas.pytorch.base_trainer.BaseTrainer
    :members:

..  autoclass:: nni.nas.pytorch.trainer.Trainer
    :members:
```

### 重新训练

```eval_rst
..  autofunction:: nni.nas.pytorch.fixed.apply_fixed_architecture

..  autoclass:: nni.nas.pytorch.fixed.FixedArchitecture
    :members:
```

### 分布式 NAS

```eval_rst
..  autofunction:: nni.nas.pytorch.classic_nas.get_and_apply_next_architecture

..  autoclass:: nni.nas.pytorch.classic_nas.mutator.ClassicMutator
    :members:
```

### 回调

```eval_rst
..  autoclass:: nni.nas.pytorch.callbacks.Callback
    :members:

..  autoclass:: nni.nas.pytorch.callbacks.LRSchedulerCallback
    :members:

..  autoclass:: nni.nas.pytorch.callbacks.ArchitectureCheckpoint
    :members:

..  autoclass:: nni.nas.pytorch.callbacks.ModelCheckpoint
    :members:
```

### 工具

```eval_rst
..  autoclass:: nni.nas.pytorch.utils.AverageMeterGroup
    :members:

..  autoclass:: nni.nas.pytorch.utils.AverageMeter
    :members:

..  autofunction:: nni.nas.pytorch.utils.to_device
```
