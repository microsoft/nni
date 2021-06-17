神经网络结构搜索在 NNI 上的应用
=======================================

.. Note:: NNI's latest NAS supports are all based on Retiarii Framework, users who are still on `early version using NNI NAS v1.0 <https://nni.readthedocs.io/en/v2.2/nas.html>`__ shall migrate your work to Retiarii as soon as possible.

.. contents::

Motivation
----------------------------------------

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 `NASNet <https://arxiv.org/abs/1707.07012>`__\ , `ENAS <https://arxiv.org/abs/1802.03268>`__\ , `DARTS <https://arxiv.org/abs/1806.09055>`__\ , `Network Morphism <https://arxiv.org/abs/1806.10282>`__\ 和 `Evolution <https://arxiv.org/abs/1703.01041>`__。 此外，新的创新不断涌现。 Recent research has proven the feasibility of automatic NAS and has led to models that beat many manually designed and tuned models. Representative works include `NASNet <https://arxiv.org/abs/1707.07012>`__\ , `ENAS <https://arxiv.org/abs/1802.03268>`__\ , `DARTS <https://arxiv.org/abs/1806.09055>`__\ , `Network Morphism <https://arxiv.org/abs/1806.10282>`__\ , and `Evolution <https://arxiv.org/abs/1703.01041>`__. In addition, new innovations continue to emerge.

However, it is pretty hard to use existing NAS work to help develop common DNN models. Therefore, we designed `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__, a novel NAS/HPO framework, and implemented it in NNI. 经典 NAS 算法的过程类似于超参调优，通过 ``nnictl`` 来启动 Experiment，每个子模型会作为 Trial 运行。 不同之处在于，搜索空间文件是通过运行 ``nnictl ss_gen``，从用户模型（已包含搜索空间）中自动生成。 下表列出了经典 NAS 模式支持的算法。 将来版本会支持更多算法。 以此为动力，NNI 的目标是提供统一的体系结构，以加速 NAS 上的创新，并将最新的算法更快地应用于现实世界中的问题上。

概述
--------

There are three key characteristics of the Retiarii framework:

* Simple APIs are provided for defining model search space within PyTorch/TensorFlow model.
为了使用 NNI NAS, 建议用户先通读这篇文档 `the tutorial of NAS API for building search space <./WriteSearchSpace.rst>`__。
* System-level optimizations are implemented for speeding up the exploration.

通过统一的接口，有两种方法来使用神经网络架构搜索。 一种称为 `one-shot NAS <#supported-one-shot-nas-algorithms>`__ ，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 `第二种 <#supported-classic-nas-algorithms>`__ 是经典的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行。 称之为经典的 NAS。 Mutli-trial NAS trains each sampled model in the model space independently, while One-shot NAS samples the model from a super model. After constructing the model space, users can use either exploration appraoch to explore the model space. 


NAS 可视化
-----------------

Multi-trial NAS means each sampled model from model space is trained independently. 一次性 NAS 可以通过可视化工具来查看。 点 `这里 <./Visualization.rst>`__ 了解更多细节。 The algorithm to sample models from model space is called exploration strategy. NNI has supported the following exploration strategies for multi-trial NAS.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 名称
     - 算法简介
   * - Random Strategy
     - 从搜索空间中随机选择模型 (``nni.retiarii.strategy.Random``)
   * - Name
     - Sampling new model(s) from user defined model space using grid search algorithm. (``nni.retiarii.strategy.GridSearch``)
   * - Regularized Evolution
     - Generating new model(s) from generated models using `regularized evolution algorithm <https://arxiv.org/abs/1802.01548>`__ . (``nni.retiarii.strategy.RegularizedEvolution``)
   * - TPE Strategy
     - Sampling new model(s) from user defined model space using `TPE algorithm <https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`__ . (``nni.retiarii.strategy.TPEStrategy``)
   * - RL Strategy
     - It uses `PPO algorithm <https://arxiv.org/abs/1707.06347>`__ to sample new model(s) from user defined model space. (``nni.retiarii.strategy.PolicyBasedRL``)


Please refer to `here <./multi_trial_nas.rst>`__ for detailed usage of multi-trial NAS.

支持的 One-shot NAS 算法
---------------------------------

One-shot NAS means building model space into a super-model, training the super-model with weight sharing, and then sampling models from the super-model to find the best one. 参考  `这里 <NasGuide.rst>`__，了解如何使用 one-shot NAS 算法。
NNI 目前支持下面列出的 One-Shot NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 `NNI API <#use-nni-api>`__， 实现其它算法，以使更多人受益。 More one-shot NAS will be supported soon.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - One-shot Algorithm Name
     - 算法简介
   * - `ENAS <ENAS.rst>`__
     - `Efficient Neural Architecture Search via Parameter Sharing <https://arxiv.org/abs/1802.03268>`__. 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 它通过在子模型间共享参数来实现加速和出色的性能指标。
   * - `DARTS <DARTS.rst>`__
     - `DARTS: Differentiable Architecture Search <https://arxiv.org/abs/1806.09055>`__ 介绍了一种用于双级优化的可区分网络体系结构搜索的新算法。
   * - `SPOS <SPOS.rst>`__
     - `Single Path One-Shot Neural Architecture Search with Uniform Sampling <https://arxiv.org/abs/1904.00420>`__ 论文构造了一个采用统一的路径采样方法来训练简化的超网络，并使用进化算法来提高搜索神经网络结构的效率。
   * - `ProxylessNAS <Proxylessnas.rst>`__
     - `ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware <https://arxiv.org/abs/1812.00332>`__. 它删除了代理，直接从大规模目标任务和目标硬件平台进行学习。

参考 `这里 <ClassicNas.rst>`__ ，了解如何使用经典 NAS 算法。

参考和反馈
----------------------

* `Quick Start <./QuickStart.rst>`__ ;
* `Construct Your Model Space <./construct_space.rst>`__ ;
* `Retiarii: A Deep Learning Exploratory-Training Framework <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ ;
* 在Github 中 `提交此功能的 Bug <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__；
* 在Github 中 `提交新功能或请求改进 <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__。
