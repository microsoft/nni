Retiarii for Neural Architecture Search
=======================================

.. toctree::
   :hidden:
   :titlesonly:

   Quick Start <../tutorials/cp_hello_nas_quickstart>
   construct_space
   exploration_strategy
   evaluator
   advanced_usage
   reference

.. attention:: NNI's latest NAS supports are all based on Retiarii Framework, users who are still on `early version using NNI NAS v1.0 <https://nni.readthedocs.io/en/v2.2/nas.html>`__ shall migrate your work to Retiarii as soon as possible.

.. note:: PyTorch is the **only supported framework on Retiarii**. Inquiries of NAS support on Tensorflow is in `this discussion <https://github.com/microsoft/nni/discussions/4605>`__. If you intend to run NAS with DL frameworks other than PyTorch and Tensorflow, please `open new issues <https://github.com/microsoft/nni/issues>`__ to let us know.

.. Using rubric to prevent the section heading to be include into toc

.. rubric:: Motivation

Automatic neural architecture search is playing an increasingly important role in finding better models. Recent research has proven the feasibility of automatic NAS and has led to models that beat many manually designed and tuned models. Representative works include `NASNet <https://arxiv.org/abs/1707.07012>`__, `ENAS <https://arxiv.org/abs/1802.03268>`__, `DARTS <https://arxiv.org/abs/1806.09055>`__, `Network Morphism <https://arxiv.org/abs/1806.10282>`__, and `Evolution <https://arxiv.org/abs/1703.01041>`__. In addition, new innovations continue to emerge.

However, it is pretty hard to use existing NAS work to help develop common DNN models. Therefore, we designed `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__, a novel NAS/HPO framework, and implemented it in NNI. It helps users easily construct a model space (or search space, tuning space), and utilize existing NAS algorithms. The framework also facilitates NAS innovation and is used to design new NAS algorithms.

In summary, we highlight the following features for Retiarii:

* Simple APIs are provided for defining model search space within a deep learning model.
* SOTA NAS algorithms are built-in to be used for exploring model search space.
* System-level optimizations are implemented for speeding up the exploration.

.. rubric:: Overview

High-level speaking, aiming to solve any particular task with neural architecture search typically requires: search space design, search strategy selection, and performance evaluation. The three components work together with the following loop (the figure is from the famous `NAS survey <https://arxiv.org/abs/1808.05377>`__):

.. image:: ../../img/nas_abstract_illustration.png

To be consistent, we will use the following terminologies throughout our documentation:

* *Model search space*: it means a set of models from which the best model is explored/searched. Sometimes we use *search space* or *model space* in short.
* *Exploration strategy*: the algorithm that is used to explore a model search space. Sometimes we also call it *search strategy*.
* *Model evaluator*: it is used to train a model and evaluate the model's performance.

Concretely, an exploration strategy selects an architecture from a predefined search space. The architecture is passed to a performance evaluation to get a score, which represents how well this architecture performs on a particular task. This process is repeated until the search process is able to find the best architecture.

During such process, we list out the core engineering challenges (which are also pointed out by the famous `NAS survey <https://arxiv.org/abs/1808.05377>`__) and the solutions NNI has provided to address them:

* **Search space design:** The search space defines which architectures can be represented in principle. Incorporating prior knowledge about typical properties of architectures well-suited for a task can reduce the size of the search space and simplify the search. However, this also introduces a human bias, which may prevent finding novel architectural building blocks that go beyond the current human knowledge. In NNI, we provide a wide range of APIs to build the search space. There are :doc:`high-level APIs <construct_space>`, that enables incorporating human knowledge about what makes a good architecture or search space. There are also :doc:`low-level APIs <mutator>`, that is a list of primitives to construct a network from operator to operator.
* **Exploration strategy:** The exploration strategy details how to explore the search space (which is often exponentially large). It encompasses the classical exploration-exploitation trade-off since, on the one hand, it is desirable to find well-performing architectures quickly, while on the other hand, premature convergence to a region of suboptimal architectures should be avoided. In NNI, we have also provided :doc:`a list of strategies <exploration_strategy>`. Some of them are powerful, but time consuming, while others might be suboptimal but really efficient. Users can always find one that matches their need.
* **Performance estimation / evaluator:** The objective of NAS is typically to find architectures that achieve high predictive performance on unseen data. Performance estimation refers to the process of estimating this performance. In NNI, this process is implemented with :doc:`evaluator <evaluator>`, which is responsible of estimating a model's performance. The choices of evaluators also range from the simplest option, e.g., to perform a standard training and validation of the architecture on data, to complex configurations and implementations.

.. rubric:: Writing Model Space

The following APIs are provided to ease the engineering effort of writing a new search space.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Category
     - Brief Description
   * - :ref:`nas-layer-choice`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Select from some PyTorch modules
   * - :ref:`nas-input-choice`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Select from some inputs (tensors)
   * - :ref:`nas-value-choice`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Select from some candidate values
   * - :ref:`nas-repeat`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Repeat a block by a variable number of times
   * - :ref:`nas-cell`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Cell structure popularly used in literature
   * - :ref:`nas-cell-101`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Cell structure (variant) proposed by NAS-Bench-101
   * - :ref:`nas-cell-201`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Cell structure (variant) proposed by NAS-Bench-201
   * - :ref:`nas-autoactivation`
     - :ref:`Hyper-modules <hyper-modules>`
     - Searching for activation functions
   * - :doc:`Mutator <mutator>`
     - :doc:`mutator`
     - Flexible mutations on graphs

.. rubric:: Exploring the Search Space

We provide the following (built-in) algorithms to explore the user-defined search space.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Category
     - Brief Description
   * - :ref:`random-strategy`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Randomly sample an architecture each time
   * - :ref:`grid-search-strategy`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Traverse the search space and try all possibilities
   * - :ref:`regularized-evolution-strategy`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Evolution algorithm for NAS. `Reference <https://arxiv.org/abs/1802.01548>`__
   * - :ref:`tpe-strategy`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Tree-structured Parzen Estimator (TPE). `Reference <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
   * - :ref:`policy-based-rl-strategy`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Policy-based reinforcement learning, based on implementation of tianshou. `Reference <https://arxiv.org/abs/1611.01578>`__
   * - :ref:`darts-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - Continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent. `Reference <https://arxiv.org/abs/1806.09055>`__
   * - :ref:`enas-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - RL controller learns to generate the best network on a super-net. `Reference <https://arxiv.org/abs/1802.03268>`__
   * - :ref:`fbnet-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - Choose the best block by using Gumbel Softmax random sampling and differentiable training. `Reference <https://arxiv.org/abs/1812.03443>`__
   * - :ref:`spos-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - Train a super-net with uniform path sampling. `Reference <https://arxiv.org/abs/1904.00420>`__
   * - :ref:`proxylessnas-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - A low-memory-consuming optimized version of differentiable architecture search. `Reference <https://arxiv.org/abs/1812.00332>`__

.. rubric:: Evaluators

The evaluator APIs can be used to build performance assessment component of your neural architecture search process.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Brief Description
   * - :ref:`functional-evaluator`
     - General
     - Evaluate with any Python function
   * - :ref:`classification-evaluator`
     - Built upon `PyTorch Lightning <https://www.pytorchlightning.ai/>`__
     - For classification tasks
   * - :ref:`regression-evaluator`
     - Built upon `PyTorch Lightning <https://www.pytorchlightning.ai/>`__
     - For regression tasks
