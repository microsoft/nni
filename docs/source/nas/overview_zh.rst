.. 48c39585a539a877461aadef63078c48

神经架构搜索
===========================

.. toctree::
   :hidden:

   快速入门 </tutorials/hello_nas>
   构建搜索空间 <construct_space>
   探索策略 <exploration_strategy>
   评估器 <evaluator>
   高级用法 <advanced_usage>

.. attention:: NNI 最新的架构搜索支持都是基于 Retiarii 框架，还在使用 `NNI 架构搜索的早期版本 <https://nni.readthedocs.io/en/v2.2/nas.html>`__ 的用户应尽快将您的工作迁移到 Retiarii。我们计划在接下来的几个版本中删除旧的架构搜索框架。

.. attention:: PyTorch 是 **Retiarii 唯一支持的框架**。有关 Tensorflow 上架构搜索支持的需求在 `此讨论 <https://github.com/microsoft/nni/discussions/4605>`__ 中。另外，如果您打算使用 PyTorch 和 Tensorflow 以外的 DL 框架运行 NAS，请 `创建新 issue <https://github.com/microsoft/nni/issues>`__ 让我们知道。

概述
------

自动神经架构搜索 (Neural Architecture Search, NAS）在寻找更好的模型方面发挥着越来越重要的作用。最近的研究证明了自动架构搜索的可行性，并导致模型击败了许多手动设计和调整的模型。其中具有代表性的有 `NASNet <https://arxiv.org/abs/1707.07012>`__、 `ENAS <https://arxiv.org/abs/1802.03268>`__、 `DARTS <https://arxiv.org/ abs/1806.09055>`__、 `Network Morphism <https://arxiv.org/abs/1806.10282>`__ 和 `进化算法 <https://arxiv.org/abs/1703.01041>`__。此外，新的创新正不断涌现。

总的来说，使用神经架构搜索解决任何特定任务通常需要：搜索空间设计、搜索策略选择和性能评估。这三个组件形成如下的循环（图来自于 `架构搜索综述 <https://arxiv.org/abs/1808.05377>`__）：

.. image:: ../../img/nas_abstract_illustration.png
   :align: center
   :width: 700

在这个图中：

* *模型搜索空间* 是指一组模型，从中探索/搜索最佳模型，简称为 *搜索空间* 或 *模型空间*。
* *探索策略* 是用于探索模型搜索空间的算法。有时我们也称它为 *搜索策略*。
* *模型评估者* 负责训练模型并评估其性能。

该过程类似于 :doc:`超参数优化 </hpo/overview>`，只不过目标是最佳网络结构而不是最优超参数。具体来说，探索策略从预定义的搜索空间中选择架构。该架构被传递给性能评估以获得评分，该评分表示这个网络结构在特定任务上的表现。重复此过程，直到搜索过程能够找到最优的网络结构。

主要特点
------------

NNI 中当前的架构搜索框架由 `Retiarii: A Deep Learning Exploratory-Training Framework <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ 的研究支撑，具有以下特点：

* :doc:`简单的 API，让您轻松构建搜索空间 <construct_space>`
* :doc:`SOTA 架构搜索算法，以高效探索搜索空间 <exploration_strategy>`
* :doc:`后端支持，在大规模 AI 平台上运行实验 </experiment/overview>`

为什么使用 NNI 的架构搜索
-------------------------------

若没有 NNI，实现架构搜索将极具挑战性，主要包含以下三个方面。当用户想在自己的场景中尝试架构搜索技术时，NNI 提供的解决方案可以极大程度上减轻用户的工作量。

搜索空间设计
^^^^^^^^^^^^^^^^^^^

搜索空间定义了架构的可行域集合。为了简化搜索，我们通常需要结合任务相关的先验知识，减小搜索空间的规模。然而，这也引入了人类的偏见，在某种程度上可能会丧失突破人类认知的可能性。无论如何，对于初学者来说，搜索空间设计是一个极具挑战性的任务，因为他们可能无法在简单的空间和丰富的想象力之间取得平衡。

在 NNI 中，我们提供了不同层级的 API 来构建搜索空间。有 :doc:`高层 API <construct_space>`，引入大量先验，帮助用户迅速了解什么是好的架构或搜索空间；也有 :doc:`底层 API <mutator>`，提供了最底层的算子和图变换原语。

探索策略
^^^^^^^^^^^^^^^^^^^^

探索策略定义了如何探索搜索空间（通常是指数级规模的）。它包含经典的探索-利用权衡。一方面，我们希望快速找到性能良好的架构；而另一方面，我们也应避免过早收敛到次优架构的区域。我们往往需要通常通过反复试验找到特定场景的“最佳”探索策略。由于许多近期发表的探索策略都是使用自己的代码库实现的，因此从一个切换到另一个变得非常麻烦。

在 NNI 中，我们还提供了 :doc:`一系列的探索策略 <exploration_strategy>`。其中一些功能强大但耗时，而另一些可能不能找到最优架构但非常高效。鉴于所有策略都使用统一的用户接口实现，用户可以轻松找到符合他们需求的策略。

性能评估
^^^^^^^^^^^^^^^^^^^^^^

架构搜索的目标通常是找到能够在测试数据集表现理想的网络结构。性能评估的作用便是量化每个网络的好坏。其主要难点在于可扩展性，即如何在大规模训练平台上同时运行和管理多个试验。

在 NNI 中，我们使用 :doc:`evaluator <evaluator>` 来标准化性能评估流程。它负责估计模型的性能。NNI 内建了不少性能评估器，从最简单的交叉验证，到复杂的自定义配置。评估器在 *试验 (trials)* 中运行，可以通过我们强大的 :doc:`训练平台 </experiment/training_service/overview>` 将试验分发到大规模训练平台上。

教程
---------

要开始使用 NNI 架构搜索框架，我们建议至少阅读以下教程：

* :doc:`快速入门 </tutorials/hello_nas>`
* :doc:`构建搜索空间 <construct_space>`
* :doc:`探索策略 <exploration_strategy>`
* :doc:`评估器 <evaluator>`

资源
---------

以下文章将有助于更好地了解 NAS 的最新发展：

* `神经架构搜索：综述 <https://arxiv.org/abs/1808.05377>`__
* `神经架构搜索的综述：挑战和解决方案 <https://arxiv.org/abs/2006.02903>`__
