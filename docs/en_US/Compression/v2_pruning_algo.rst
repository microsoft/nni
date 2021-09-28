Supported Pruning Algorithms in NNI
===================================

NNI provides several pruning algorithms that reproducing from the papers. In pruning v2, NNI split the pruning algorithm into more detailed components.
This means users can freely combine components from different algorithms,
or easily use a component of their own implementation to replace a step in the original algorithm to implement their own pruning algorithm.

Right now, pruning algorithms with how to generate masks in one step are implemented as pruners, and how to schedule sparsity in each iteration are implemented as task generators.

** Pruner **

* `Level Pruner <#level-pruner>`__
* `L1 Norm Pruner <#l1-norm-pruner>`__
* `L2 Norm Pruner <#l2-norm-pruner>`__
* `FPGM Pruner <#fpgm-pruner>`__
* `Slim Pruner <#slim-pruner>`__
* `Activation APoZ Rank Pruner <#activation-apoz-rank-pruner>`__
* `Activation Mean Rank Pruner <#activation-mean-rank-pruner>`__
* `Taylor FO Weight Pruner <#taylor-fo-weight-pruner>`__
* `ADMM Pruner <#admm-pruner>`__

** Task Generator **

* `Linear Task Generator <#linear-task-generator>`__
* `AGP Task Generator <#agp-task-generator>`__
* `Lottery Ticket Task Generator <#lottery-ticket-task-generator>`__
* `Simulated Annealing Task Generator <#simulated-annealing-task-generator>`__

Level Pruner
------------

This is a basic pruner, and in some papers called it magnitude pruning or fine-grained pruning.
It will mask the weight in each specified layer with smaller absolute value by a ratio configured in the config list.

Useage
^^^^^^

User configuration for Level Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

..  autoclass:: nni.algorithms.compression.v2.pytorch.pruning.LevelPruner
