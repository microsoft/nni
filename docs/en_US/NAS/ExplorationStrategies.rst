Exploration Strategies for Multi-trial NAS
==========================================

NNI provides the following exploration strategies for multi-trial NAS. Users could also customize new exploration strategies.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - `Random Strategy <./ApiReference.rst#nni.retiarii.strategy.Random>`__
     - Randomly sampling new model(s) from user defined model space. (``nni.retiarii.strategy.Random``)
   * - `Grid Search <./ApiReference.rst#nni.retiarii.strategy.GridSearch>`__
     - Sampling new model(s) from user defined model space using grid search algorithm. (``nni.retiarii.strategy.GridSearch``)
   * - `Regularized Evolution <./ApiReference.rst#nni.retiarii.strategy.RegularizedEvolution>`__
     - Generating new model(s) from generated models using `regularized evolution algorithm <https://arxiv.org/abs/1802.01548>`__ . (``nni.retiarii.strategy.RegularizedEvolution``)
   * - `TPE Strategy <./ApiReference.rst#nni.retiarii.strategy.TPEStrategy>`__
     - Sampling new model(s) from user defined model space using `TPE algorithm <https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`__ . (``nni.retiarii.strategy.TPEStrategy``)
   * - `RL Strategy <./ApiReference.rst#nni.retiarii.strategy.PolicyBasedRL>`__
     - It uses `PPO algorithm <https://arxiv.org/abs/1707.06347>`__ to sample new model(s) from user defined model space. (``nni.retiarii.strategy.PolicyBasedRL``)