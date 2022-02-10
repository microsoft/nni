Retiarii for Neural Architecture Search
=======================================

.. attention:: NNI's latest NAS supports are all based on Retiarii Framework, users who are still on `early version using NNI NAS v1.0 <https://nni.readthedocs.io/en/v2.2/nas.html>`__ shall migrate your work to Retiarii as soon as possible.

.. contents::

Motivation
----------

Automatic neural architecture search is playing an increasingly important role in finding better models. Recent research has proven the feasibility of automatic NAS and has led to models that beat many manually designed and tuned models. Representative works include `NASNet <https://arxiv.org/abs/1707.07012>`__\ , `ENAS <https://arxiv.org/abs/1802.03268>`__\ , `DARTS <https://arxiv.org/abs/1806.09055>`__\ , `Network Morphism <https://arxiv.org/abs/1806.10282>`__\ , and `Evolution <https://arxiv.org/abs/1703.01041>`__. In addition, new innovations continue to emerge.

However, it is pretty hard to use existing NAS work to help develop common DNN models. Therefore, we designed `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__, a novel NAS/HPO framework, and implemented it in NNI. It helps users easily construct a model space (or search space, tuning space), and utilize existing NAS algorithms. The framework also facilitates NAS innovation and is used to design new NAS algorithms.

Overview
--------

There are three key characteristics of the Retiarii framework:

* Simple APIs are provided for defining model search space within PyTorch/TensorFlow model.
* SOTA NAS algorithms are built-in to be used for exploring model search space.
* System-level optimizations are implemented for speeding up the exploration.

There are two types of model space exploration approach: **Multi-trial NAS** and **One-shot NAS**. Mutli-trial NAS trains each sampled model in the model space independently, while One-shot NAS samples the model from a super model. After constructing the model space, users can use either exploration appraoch to explore the model space. 


Multi-trial NAS
---------------

Multi-trial NAS means each sampled model from model space is trained independently. A typical multi-trial NAS is `NASNet <https://arxiv.org/abs/1707.07012>`__. The algorithm to sample models from model space is called exploration strategy. NNI has supported the following exploration strategies for multi-trial NAS.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Exploration Strategy Name
     - Brief Introduction of Algorithm
   * - Random Strategy
     - Randomly sampling new model(s) from user defined model space. (``nni.retiarii.strategy.Random``)
   * - Grid Search
     - Sampling new model(s) from user defined model space using grid search algorithm. (``nni.retiarii.strategy.GridSearch``)
   * - Regularized Evolution
     - Generating new model(s) from generated models using `regularized evolution algorithm <https://arxiv.org/abs/1802.01548>`__ . (``nni.retiarii.strategy.RegularizedEvolution``)
   * - TPE Strategy
     - Sampling new model(s) from user defined model space using `TPE algorithm <https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`__ . (``nni.retiarii.strategy.TPEStrategy``)
   * - RL Strategy
     - It uses `PPO algorithm <https://arxiv.org/abs/1707.06347>`__ to sample new model(s) from user defined model space. (``nni.retiarii.strategy.PolicyBasedRL``)


Please refer to `here <./multi_trial_nas.rst>`__ for detailed usage of multi-trial NAS.

One-shot NAS
------------

One-shot NAS means building model space into a super-model, training the super-model with weight sharing, and then sampling models from the super-model to find the best one. `DARTS <https://arxiv.org/abs/1806.09055>`__ is a typical one-shot NAS.
Below is the supported one-shot NAS algorithms. More one-shot NAS will be supported soon.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - One-shot Algorithm Name
     - Brief Introduction of Algorithm
   * - `ENAS <ENAS.rst>`__
     - `Efficient Neural Architecture Search via Parameter Sharing <https://arxiv.org/abs/1802.03268>`__. In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. It uses parameter sharing between child models to achieve fast speed and excellent performance.
   * - `DARTS <DARTS.rst>`__
     - `DARTS: Differentiable Architecture Search <https://arxiv.org/abs/1806.09055>`__ introduces a novel algorithm for differentiable network architecture search on bilevel optimization.
   * - `SPOS <SPOS.rst>`__
     - `Single Path One-Shot Neural Architecture Search with Uniform Sampling <https://arxiv.org/abs/1904.00420>`__ constructs a simplified supernet trained with a uniform path sampling method and applies an evolutionary algorithm to efficiently search for the best-performing architectures.
   * - `ProxylessNAS <Proxylessnas.rst>`__
     - `ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware <https://arxiv.org/abs/1812.00332>`__. It removes proxy, directly learns the architectures for large-scale target tasks and target hardware platforms.

Please refer to `here <one_shot_nas.rst>`__ for detailed usage of one-shot NAS algorithms.

Reference and Feedback
----------------------

* `Quick Start <./QuickStart.rst>`__ ;
* `Construct Your Model Space <./construct_space.rst>`__ ;
* `Retiarii: A Deep Learning Exploratory-Training Framework <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ ;
* To `report a bug <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__ for this feature in GitHub ;
* To `file a feature or improvement request <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__ for this feature in GitHub .
