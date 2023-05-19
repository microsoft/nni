Pruning Algorithm Supported in NNI
==================================

Note that not all pruners from the previous version have been migrated to the new framework yet.
NNI has plans to migrate all pruners that were implemented in NNI 3.2.

If you believe that a certain old pruner has not been implemented or that another pruning algorithm would be valuable,
please feel free to contact us. We will prioritize and expedite support accordingly.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - :ref:`new-level-pruner`
     - Pruning the specified ratio on each weight element based on absolute value of weight element
   * - :ref:`new-l1-norm-pruner`
     - Pruning output channels with the smallest L1 norm of weights (Pruning Filters for Efficient Convnets) `Reference Paper <https://arxiv.org/abs/1608.08710>`__
   * - :ref:`new-l2-norm-pruner`
     - Pruning output channels with the smallest L2 norm of weights
   * - :ref:`new-fpgm-pruner`
     - Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration `Reference Paper <https://arxiv.org/abs/1811.00250>`__
   * - :ref:`new-slim-pruner`
     - Pruning output channels by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) `Reference Paper <https://arxiv.org/abs/1708.06519>`__
   * - :ref:`new-taylor-pruner`
     - Pruning filters based on the first order taylor expansion on weights(Importance Estimation for Neural Network Pruning) `Reference Paper <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__
   * - :ref:`new-linear-pruner`
     - Sparsity ratio increases linearly during each pruning rounds, in each round, using a basic pruner to prune the model.
   * - :ref:`new-agp-pruner`
     - Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) `Reference Paper <https://arxiv.org/abs/1710.01878>`__
   * - :ref:`new-movement-pruner`
     - Movement Pruning: Adaptive Sparsity by Fine-Tuning `Reference Paper <https://arxiv.org/abs/2005.07683>`__
