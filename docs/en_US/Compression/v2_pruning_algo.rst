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

User configuration for L1 Norm Pruner
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

User configuration for L2 Norm Pruner
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

User configuration for FPGM Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.FPGMPruner

Slim Pruner
-----------

Slim pruner adds sparsity regularization on the scaling factors of batch normalization (BN) layers during training to identify unimportant channels.
The channels with small scaling factor values will be pruned.

For more details, please refer to `'Learning Efficient Convolutional Networks through Network Slimming' <https://arxiv.org/pdf/1708.06519.pdf>`__\.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import SlimPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
   pruner = SlimPruner(model, config_list, trainer, optimizer, criterion, training_epochs=1)
   masked_model, masks = pruner.compress()

User configuration for Slim Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.SlimPruner

Activation APoZ Rank Pruner
---------------------------

Activation APoZ rank pruner is a pruner which prunes on the first weight dimension,
with the smallest importance criterion ``APoZ`` calculated from the output activations of convolution layers to achieve a preset level of network sparsity.
The pruning criterion ``APoZ`` is explained in the paper `Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures <https://arxiv.org/abs/1607.03250>`__.

The APoZ is defined as:

:math:`APoZ_{c}^{(i)} = APoZ\left(O_{c}^{(i)}\right)=\frac{\sum_{k}^{N} \sum_{j}^{M} f\left(O_{c, j}^{(i)}(k)=0\right)}{N \times M}`

Activation APoZ rank pruner also supports dependency-aware mode.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import ActivationAPoZRankPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
   pruner = ActivationAPoZRankPruner(model, config_list, trainer, optimizer, criterion, training_batches=20)
   masked_model, masks = pruner.compress()

User configuration for Activation APoZ Rank Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.ActivationAPoZRankPruner

Activation Mean Rank Pruner
---------------------------

Activation mean rank pruner is a pruner which prunes on the first weight dimension,
with the smallest importance criterion ``mean activation`` calculated from the output activations of convolution layers to achieve a preset level of network sparsity.
The pruning criterion ``mean activation`` is explained in section 2.2 of the paper `Pruning Convolutional Neural Networks for Resource Efficient Inference <https://arxiv.org/abs/1611.06440>`__.

Activation mean rank pruner also supports dependency-aware mode.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import ActivationMeanRankPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
   pruner = ActivationMeanRankPruner(model, config_list, trainer, optimizer, criterion, training_batches=20)
   masked_model, masks = pruner.compress()

User configuration for Activation Mean Rank Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.ActivationMeanRankPruner

Taylor FO Weight Pruner
-----------------------

Taylor FO weight pruner is a pruner which prunes on the first weight dimension,
based on estimated importance calculated from the first order taylor expansion on weights to achieve a preset level of network sparsity.
The estimated importance is defined as the paper `Importance Estimation for Neural Network Pruning <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__.

:math:`\widehat{\mathcal{I}}_{\mathcal{S}}^{(1)}(\mathbf{W}) \triangleq \sum_{s \in \mathcal{S}} \mathcal{I}_{s}^{(1)}(\mathbf{W})=\sum_{s \in \mathcal{S}}\left(g_{s} w_{s}\right)^{2}`

Taylor FO weight pruner also supports dependency-aware mode.

What's more, we provide a global-sort mode for this pruner which is aligned with paper implementation.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import TaylorFOWeightPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
   pruner = TaylorFOWeightPruner(model, config_list, trainer, optimizer, criterion, training_batches=20)
   masked_model, masks = pruner.compress()

User configuration for Activation Mean Rank Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.TaylorFOWeightPruner

ADMM Pruner
-----------

Alternating Direction Method of Multipliers (ADMM) is a mathematical optimization technique,
by decomposing the original nonconvex problem into two subproblems that can be solved iteratively.
In weight pruning problem, these two subproblems are solved via 1) gradient descent algorithm and 2) Euclidean projection respectively. 

During the process of solving these two subproblems, the weights of the original model will be changed.
An one-shot pruner will then be applied to prune the model according to the config list given.

This solution framework applies both to non-structured and different variations of structured pruning schemes.

For more details, please refer to `A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers <https://arxiv.org/abs/1804.03294>`__.

Useage
^^^^^^

.. code-block:: python

   from nni.algorithms.compression.v2.pytorch.pruning import ADMMPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
   pruner = ADMMPruner(model, config_list, trainer, optimizer, criterion, training_batches=20)
   masked_model, masks = pruner.compress()

User configuration for ADMM Pruner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

.. autoclass:: nni.algorithms.compression.v2.pytorch.pruning.ADMMPruner
