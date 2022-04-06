Pruner in NNI
=============

Pruning algorithms compress the original network by removing redundant weights or channels of layers, which can reduce model complexity and mitigate the over-fitting issue.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - :ref:`level-pruner`
     - Pruning the specified ratio on each weight element based on absolute value of weight element
   * - :ref:`l1-norm-pruner`
     - Pruning output channels with the smallest L1 norm of weights (Pruning Filters for Efficient Convnets) `Reference Paper <https://arxiv.org/abs/1608.08710>`__
   * - :ref:`l2-norm-pruner`
     - Pruning output channels with the smallest L2 norm of weights
   * - :ref:`fpgm-pruner`
     - Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration `Reference Paper <https://arxiv.org/abs/1811.00250>`__
   * - :ref:`slim-pruner`
     - Pruning output channels by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) `Reference Paper <https://arxiv.org/abs/1708.06519>`__
   * - :ref:`activation-apoz-rank-pruner`
     - Pruning output channels based on the metric APoZ (average percentage of zeros) which measures the percentage of zeros in activations of (convolutional) layers. `Reference Paper <https://arxiv.org/abs/1607.03250>`__
   * - :ref:`activation-mean-rank-pruner`
     - Pruning output channels based on the metric that calculates the smallest mean value of output activations
   * - :ref:`taylor-fo-weight-pruner`
     - Pruning filters based on the first order taylor expansion on weights(Importance Estimation for Neural Network Pruning) `Reference Paper <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__
   * - :ref:`admm-pruner`
     - Pruning based on ADMM optimization technique `Reference Paper <https://arxiv.org/abs/1804.03294>`__
   * - :ref:`linear-pruner`
     - Sparsity ratio increases linearly during each pruning rounds, in each round, using a basic pruner to prune the model.
   * - :ref:`agp-pruner`
     - Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) `Reference Paper <https://arxiv.org/abs/1710.01878>`__
   * - :ref:`lottery-ticket-pruner`
     - The pruning process used by "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". It prunes a model iteratively. `Reference Paper <https://arxiv.org/abs/1803.03635>`__
   * - :ref:`simulated-annealing-pruner`
     - Automatic pruning with a guided heuristic search method, Simulated Annealing algorithm `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - :ref:`auto-compress-pruner`
     - Automatic pruning by iteratively call SimulatedAnnealing Pruner and ADMM Pruner `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - :ref:`amc-pruner`
     - AMC: AutoML for Model Compression and Acceleration on Mobile Devices `Reference Paper <https://arxiv.org/abs/1802.03494>`__
   * - :ref:`movement-pruner`
     - Movement Pruning: Adaptive Sparsity by Fine-Tuning `Reference Paper <https://arxiv.org/abs/2005.07683>`__
