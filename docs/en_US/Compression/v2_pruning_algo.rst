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

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import LevelPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
   pruner = LevelPruner(model, config_list)
   masked_model, masks = pruner.compress()

User configuration for Level Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.LevelPruner

L1 Norm Pruner
--------------

L1 norm pruner compute the l1 norm of the layer weight on the first dimension,
then prune the weight blocks on this dimension with smaller l1 norm values.
i.e., compute the l1 norm of the filters in convolution layer as metric values,
compute the l1 norm of the weight by rows in linear layer as metric values.

For more details, please refer to `PRUNING FILTERS FOR EFFICIENT CONVNETS <https://arxiv.org/abs/1608.08710>`__\.

In addition, L1 norm pruner also supports dependency-aware mode.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
   pruner = L1NormPruner(model, config_list)
   masked_model, masks = pruner.compress()

User configuration for Level Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.L1NormPruner

L2 Norm Pruner
--------------

L2 norm pruner is a variant of L1 norm pruner. It uses l2 norm as metric to determine which weight elements should be pruned.

L2 norm pruner also supports dependency-aware mode.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import L2NormPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
   pruner = L2NormPruner(model, config_list)
   masked_model, masks = pruner.compress()

User configuration for Level Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.L2NormPruner

FPGM Pruner
-----------

FPGM pruner prunes the blocks of the weight on the first dimension with the smallest geometric median.
FPGM chooses the weight blocks with the most replaceable contribution.

For more details, please refer to `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/pdf/1811.00250.pdf>`__.

FPGM pruner also supports dependency-aware mode.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
   pruner = FPGMPruner(model, config_list)
   masked_model, masks = pruner.compress()

User configuration for Level Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.FPGMPruner

Slim Pruner
-----------

Slim pruner

For more details, please refer to `'Learning Efficient Convolutional Networks through Network Slimming' <https://arxiv.org/pdf/1708.06519.pdf>`__\.
