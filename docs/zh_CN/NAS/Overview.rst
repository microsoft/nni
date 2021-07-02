Retiarii 用于神经网络架构搜索
=======================================

.. Note:: NNI 最新的 NAS 支持都是基于 Retiarii 框架的，仍在使用早期版本 `NNI NAS v1.0 <https://nni.readthedocs.io/zh/v2.2/nas.html>`__ 的用户应尽快将工作迁移到 Retiarii 框架。

.. contents::

动机
----------------------------------------

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。  最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 `NASNet <https://arxiv.org/abs/1707.07012>`__\ , `ENAS <https://arxiv.org/abs/1802.03268>`__\ , `DARTS <https://arxiv.org/abs/1806.09055>`__\ , `Network Morphism <https://arxiv.org/abs/1806.10282>`__\ 和 `Evolution <https://arxiv.org/abs/1703.01041>`__。 此外，新的创新不断涌现。

然而，使用现有的 NAS 工作来帮助开发通用的 DNN 模型是相当困难的。 因此，我们设计了 `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__，一个全新的 NAS/HPO 框架，并在 NNI 中实施。 它可以帮助用户轻松构建模型空间（或搜索空间，调优空间），并利用现有的 NAS 算法。 该框架还有助于 NAS 创新，用于设计新的 NAS 算法。

概述
--------

Retiarii 框架有三个主要特点：

* 提供简单的 API，用于在 PyTorch/TensorFlow 模型中定义模型搜索空间。
* 内置前沿 NAS 算法，用于探索模型搜索空间。
* 实施系统级优化以加快探索。

有两种类型的模型空间探索方法：**Multi-trial NAS** 和 **One-shot NAS**。 Mutli-trial NAS 在模型空间中独立训练每个采样模型，而 One-shot NAS 则从一个超级模型中采样。 构建模型空间后，用户可以使用探索方法来探索模型空间。 


Multi-trial NAS
-----------------

Multi-trial NAS 意味着每个来自模型空间的抽样模型都是独立训练的。 一个典型的 multi-trial NAS 是 `NASNet <https://arxiv.org/abs/1707.07012>`__。 从模型空间中抽取模型的算法被称为探索策略。 NNI支持以下 multi-trial NAS 的探索策略。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 探索策略名称
     - 算法简介
   * - 随机策略
     - 从搜索空间中随机选择模型 (``nni.retiarii.strategy.Random``)
   * - 网格搜索
     - 使用网格搜索算法从用户定义的模型空间中采样新模型。 (``nni.retiarii.strategy.GridSearch``)
   * - 正则进化
     - 使用 `正则进化算法 <https://arxiv.org/abs/1802.01548>`__ 从生成的模型中生成新模型 (``nni.retiarii.strategy.RegularizedEvolution``)
   * - TPE 策略
     - 使用 `TPE 算法 <https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`__ 从用户定义的模型空间中生成新模型 (``nni.retiarii.strategy.TPEStrategy``)
   * - RL 策略
     - 使用 `PPO 算法 <https://arxiv.org/abs/1707.06347>`__ 从用户定义的模型空间中生成新模型 (``nni.retiarii.strategy.PolicyBasedRL``)


参考 `这里 <./multi_trial_nas.rst>`__ 获取 multi-trial NAS 详细用法。

One-shot NAS
---------------------------------

One-Shot NAS意味着将模型空间构建成一个超级模型，用权重共享的方式训练超级模型，然后从超级模型中不断采样，找到最佳模型。 `DARTS <https://arxiv.org/abs/1806.09055>`__ 是一个典型的 One-Shot NAS。
以下是已经支持的 One-Shot NAS 算法。 未来将支持更多 One-Shot NAS 算法。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - One-shot 算法名称
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

* `快速入门 <./QuickStart.rst>`__ ;
* `构建模型空间 <./construct_space.rst>`__ ;
* `Retiarii: 一个探索性的深度学习框架 <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ ;
* 在 Github 中 `提交 Bug 报告 <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__；
* 在Github 中 `提交新功能或请求改进 <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__。
