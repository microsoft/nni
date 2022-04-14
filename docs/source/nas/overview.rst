Overview
========

.. attention:: NNI's latest NAS supports are all based on Retiarii Framework, users who are still on `early version using NNI NAS v1.0 <https://nni.readthedocs.io/en/v2.2/nas.html>`__ shall migrate your work to Retiarii as soon as possible. We plan to remove the legacy NAS framework in the next few releases.

.. note:: PyTorch is the **only supported framework on Retiarii**. Inquiries of NAS support on Tensorflow is in `this discussion <https://github.com/microsoft/nni/discussions/4605>`__. If you intend to run NAS with DL frameworks other than PyTorch and Tensorflow, please `open new issues <https://github.com/microsoft/nni/issues>`__ to let us know.

Basics
------

Automatic neural architecture search is playing an increasingly important role in finding better models. Recent research has proven the feasibility of automatic NAS and has led to models that beat many manually designed and tuned models. Representative works include `NASNet <https://arxiv.org/abs/1707.07012>`__, `ENAS <https://arxiv.org/abs/1802.03268>`__, `DARTS <https://arxiv.org/abs/1806.09055>`__, `Network Morphism <https://arxiv.org/abs/1806.10282>`__, and `Evolution <https://arxiv.org/abs/1703.01041>`__. In addition, new innovations continue to emerge.

High-level speaking, aiming to solve any particular task with neural architecture search typically requires: search space design, search strategy selection, and performance evaluation. The three components work together with the following loop (from the famous `NAS survey <https://arxiv.org/abs/1808.05377>`__):

.. image:: ../../img/nas_abstract_illustration.png
   :align: center
   :width: 700

In this figure:

* *Model search space*  means a set of models from which the best model is explored/searched. Sometimes we use *search space* or *model space* in short.
* *Exploration strategy* is the algorithm that is used to explore a model search space. Sometimes we also call it *search strategy*.
* *Model evaluator* is responsible for training a model and evaluating its performance.

The process is similar to :doc:`Hyperparameter Optimization </hpo/overview>`, except that the target is the best architecture rather than hyperparameter. Concretely, an exploration strategy selects an architecture from a predefined search space. The architecture is passed to a performance evaluation to get a score, which represents how well this architecture performs on a particular task. This process is repeated until the search process is able to find the best architecture.

Key Features
------------

The current NAS framework in NNI is powered by the research of `Retiarii: A Deep Learning Exploratory-Training Framework <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__, where we highlight the following features:

* :doc:`Simple APIs to construct search space easily <construct_space>`
* :doc:`SOTA NAS algorithms to explore search space <exploration_strategy>`
* :doc:`Experiment backend support to scale up experiments on large-scale AI platforms </experiment/overview>`

Why NAS with NNI
----------------

We list out the three perspectives where NAS can be particularly challegning without NNI. NNI provides solutions to relieve users' engineering effort when they want to try NAS techniques in their own scenario.

Search Space Design
^^^^^^^^^^^^^^^^^^^

The search space defines which architectures can be represented in principle. Incorporating prior knowledge about typical properties of architectures well-suited for a task can reduce the size of the search space and simplify the search. However, this also introduces a human bias, which may prevent finding novel architectural building blocks that go beyond the current human knowledge. Search space design can be very challenging for beginners, who might not possess the experience to balance the richness and simplicity.

In NNI, we provide a wide range of APIs to build the search space. There are :doc:`high-level APIs <construct_space>`, that enables the possibility to incorporate human knowledge about what makes a good architecture or search space. There are also :doc:`low-level APIs <mutator>`, that is a list of primitives to construct a network from operation to operation.

Exploration strategy
^^^^^^^^^^^^^^^^^^^^

The exploration strategy details how to explore the search space (which is often exponentially large). It encompasses the classical exploration-exploitation trade-off since, on the one hand, it is desirable to find well-performing architectures quickly, while on the other hand, premature convergence to a region of suboptimal architectures should be avoided. The "best" exploration strategy for a particular scenario is usually found via trial-and-error. As many state-of-the-art strategies are implemented with their own code-base, it becomes very troublesome to switch from one to another.

In NNI, we have also provided :doc:`a list of strategies <exploration_strategy>`. Some of them are powerful yet time consuming, while others might be suboptimal but really efficient. Given that all strategies are implemented with a unified interface, users can always find one that matches their need.

Performance estimation
^^^^^^^^^^^^^^^^^^^^^^

The objective of NAS is typically to find architectures that achieve high predictive performance on unseen data. Performance estimation refers to the process of estimating this performance. The problem with performance estimation is mostly its scalability, i.e., how can I run and manage multiple trials simultaneously.

In NNI, we standardize this process is implemented with :doc:`evaluator <evaluator>`, which is responsible of estimating a model's performance. NNI has quite a few built-in supports of evaluators, ranging from the simplest option, e.g., to perform a standard training and validation of the architecture on data, to complex configurations and implementations. Evaluators are run in *trials*, where trials can be spawn onto distributed platforms with our powerful :doc:`training service </experiment/training_service/overview>`.

Tutorials
---------

To start using NNI NAS framework, we recommend at least going through the following tutorials:

* :doc:`Quickstart </tutorials/hello_nas>`
* :doc:`construct_space`
* :doc:`exploration_strategy`
* :doc:`evaluator`

Resources
---------

The following articles will help with a better understanding of the current arts of NAS:

* `Neural Architecture Search: A Survey <https://arxiv.org/abs/1808.05377>`__
* `A Comprehensive Survey of Neural Architecture Search: Challenges and Solutions <https://arxiv.org/abs/2006.02903>`__
