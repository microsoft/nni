# NAS Reference

```eval_rst
.. contents::
```

## Mutables

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

### Utilities

```eval_rst
..  autofunction:: nni.nas.pytorch.utils.global_mutable_counting
```

## Mutators

```eval_rst
..  autoclass:: nni.nas.pytorch.base_mutator.BaseMutator
    :members:

..  autoclass:: nni.nas.pytorch.mutator.Mutator
    :members:
```

### Random Mutator

```eval_rst
..  autoclass:: nni.algorithms.nas.pytorch.random.RandomMutator
    :members:
```

### Utilities

```eval_rst
..  autoclass:: nni.nas.pytorch.utils.StructuredMutableTreeNode
    :members:
```

## Trainers

### Trainer

```eval_rst
..  autoclass:: nni.nas.pytorch.base_trainer.BaseTrainer
    :members:

..  autoclass:: nni.nas.pytorch.trainer.Trainer
    :members:
```

### Retrain

```eval_rst
..  autofunction:: nni.nas.pytorch.fixed.apply_fixed_architecture

..  autoclass:: nni.nas.pytorch.fixed.FixedArchitecture
    :members:
```

### Distributed NAS

```eval_rst
..  autofunction:: nni.algorithms.nas.pytorch.classic_nas.get_and_apply_next_architecture

..  autoclass:: nni.algorithms.nas.pytorch.classic_nas.mutator.ClassicMutator
    :members:
```

### Callbacks

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

### Utilities

```eval_rst
..  autoclass:: nni.nas.pytorch.utils.AverageMeterGroup
    :members:

..  autoclass:: nni.nas.pytorch.utils.AverageMeter
    :members:

..  autofunction:: nni.nas.pytorch.utils.to_device
```
