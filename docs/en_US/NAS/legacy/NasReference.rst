NAS Reference
=============

.. contents::

Mutables
--------

..  autoclass:: nni.nas.pytorch.mutables.Mutable
    :members:

..  autoclass:: nni.nas.pytorch.mutables.LayerChoice
    :members:

..  autoclass:: nni.nas.pytorch.mutables.InputChoice
    :members:

..  autoclass:: nni.nas.pytorch.mutables.MutableScope
    :members:

Utilities
^^^^^^^^^

..  autofunction:: nni.nas.pytorch.utils.global_mutable_counting

Mutators
--------

..  autoclass:: nni.nas.pytorch.base_mutator.BaseMutator
    :members:

..  autoclass:: nni.nas.pytorch.mutator.Mutator
    :members:

Random Mutator
^^^^^^^^^^^^^^

..  autoclass:: nni.algorithms.nas.pytorch.random.RandomMutator
    :members:

Utilities
^^^^^^^^^

..  autoclass:: nni.nas.pytorch.utils.StructuredMutableTreeNode
    :members:

Trainers
--------

Trainer
^^^^^^^

..  autoclass:: nni.nas.pytorch.base_trainer.BaseTrainer
    :members:

..  autoclass:: nni.nas.pytorch.trainer.Trainer
    :members:

Retrain
^^^^^^^

..  autofunction:: nni.nas.pytorch.fixed.apply_fixed_architecture

..  autoclass:: nni.nas.pytorch.fixed.FixedArchitecture
    :members:

Distributed NAS
^^^^^^^^^^^^^^^

..  autofunction:: nni.algorithms.nas.pytorch.classic_nas.get_and_apply_next_architecture

..  autoclass:: nni.algorithms.nas.pytorch.classic_nas.mutator.ClassicMutator
    :members:

Callbacks
^^^^^^^^^

..  autoclass:: nni.nas.pytorch.callbacks.Callback
    :members:

..  autoclass:: nni.nas.pytorch.callbacks.LRSchedulerCallback
    :members:

..  autoclass:: nni.nas.pytorch.callbacks.ArchitectureCheckpoint
    :members:

..  autoclass:: nni.nas.pytorch.callbacks.ModelCheckpoint
    :members:

Utilities
^^^^^^^^^

..  autoclass:: nni.nas.pytorch.utils.AverageMeterGroup
    :members:

..  autoclass:: nni.nas.pytorch.utils.AverageMeter
    :members:

..  autofunction:: nni.nas.pytorch.utils.to_device
