Retiarii for Neural Architecture Search
=======================================

.. attention:: NNI's latest NAS supports are all based on Retiarii Framework, users who are still on `early version using NNI NAS v1.0 <https://nni.readthedocs.io/en/v2.2/nas.html>`__ shall migrate your work to Retiarii as soon as possible.

Motivation
----------

Automatic neural architecture search is playing an increasingly important role in finding better models. Recent research has proven the feasibility of automatic NAS and has led to models that beat many manually designed and tuned models. Representative works include `NASNet <https://arxiv.org/abs/1707.07012>`__, `ENAS <https://arxiv.org/abs/1802.03268>`__, `DARTS <https://arxiv.org/abs/1806.09055>`__, `Network Morphism <https://arxiv.org/abs/1806.10282>`__, and `Evolution <https://arxiv.org/abs/1703.01041>`__. In addition, new innovations continue to emerge.

However, it is pretty hard to use existing NAS work to help develop common DNN models. Therefore, we designed `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__, a novel NAS/HPO framework, and implemented it in NNI. It helps users easily construct a model space (or search space, tuning space), and utilize existing NAS algorithms. The framework also facilitates NAS innovation and is used to design new NAS algorithms.

In summary, we highlight the following features for Retiarii:

* Simple APIs are provided for defining model search space within PyTorch/TensorFlow model.
* SOTA NAS algorithms are built-in to be used for exploring model search space.
* System-level optimizations are implemented for speeding up the exploration.

Overview
--------

High-level speaking, aiming to solve any particular task with neural architecture search typically requires: search space design, search strategy selection, and performance evaluation. The three components work together with the following loop (the figure is from the famous `NAS survey <https://arxiv.org/abs/1808.05377>`__):

.. image:: ../../img/nas_abstract_illustration.png

Concretely, A search strategy selects an architecture from a predefined search space. The architecture is passed to a performance evaluation to get a score, which represents how well this architecture performs on a particular task. We iterate this process until the search process is able to find the best architecture.

During such process, we list out the core engineering challenges (which are also pointed out by the famous `NAS survey <https://arxiv.org/abs/1808.05377>`__) and the solutions NNI has provided to tackling them:

* **Search space design:** The search space defines which architectures can be represented in principle. Incorporating prior knowledge about typical properties of architectures well-suited for a task can reduce the size of the search space and simplify the search. However, this also introduces a human bias, which may prevent finding novel architectural building blocks that go beyond the current human knowledge. In NNI, we provide a wide range of APIs to build the search space. There are high-level APIs, that incorporate human knowledge about what makes a good architecture or search space. There are also low-level APIs, that is a list of primitives to construct a network from operator to operator.
* **Search strategy:**: The search strategy details how to explore the search space (which is often exponentially large). It encompasses the classical exploration-exploitation trade-off since, on the one hand, it is desirable to find well-performing architectures quickly, while on the other hand, premature convergence to a region of suboptimal architectures should be avoided. In NNI, we have also provided a list of strategies. Some of them are powerful, but time consuming, while others might be suboptimal but really efficient. Users can always find one that matches their need.
* **Performance estimation / evaluator**: The objective of NAS is typically to find architectures that achieve high predictive performance on unseen data. Performance estimation refers to the process of estimating this performance. In NNI, this process is implemented with *evaluator*, which is responsible of estimating a model's performance. The simplest option is to perform a standard training and validation of the architecture on data. But this is computationally expensive. More efficient evaluators can be implemented with our provided interface.

Writing Search Space
--------------------

The following APIs are provided for writing a new search space.





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
