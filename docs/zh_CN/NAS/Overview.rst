神经网络结构搜索在 NNI 上的应用
=======================================

.. contents::

概述
--------

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 `NASNet <https://arxiv.org/abs/1707.07012>`__\ , `ENAS <https://arxiv.org/abs/1802.03268>`__\ , `DARTS <https://arxiv.org/abs/1806.09055>`__\ , `Network Morphism <https://arxiv.org/abs/1806.10282>`__\ 和 `Evolution <https://arxiv.org/abs/1703.01041>`__。 此外，新的创新不断涌现。

但是，要实现 NAS 算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速 NAS 上的创新，并将最新的算法更快地应用于现实世界中的问题上。

通过统一的接口，有两种方法来使用神经网络架构搜索。 一种称为 `one-shot NAS <#supported-one-shot-nas-algorithms>`__ ，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 `第二种 <#supported-classic-nas-algorithms>`__ 是经典的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行。 称之为经典的 NAS。

NNI 还提供了专门的  `可视化工具 <#nas-visualization>`__，用于查看神经网络架构搜索的过程。

支持的经典 NAS 算法
--------------------------------

经典 NAS 算法的过程类似于超参调优，通过 ``nnictl`` 来启动 Experiment，每个子模型会作为 Trial 运行。 不同之处在于，搜索空间文件是通过运行 ``nnictl ss_gen``，从用户模型（已包含搜索空间）中自动生成。 下表列出了经典 NAS 模式支持的算法。 将来版本会支持更多算法。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 名称
     - 算法简介
   * - :githublink:`Random Search <examples/tuners/random_nas_tuner>`
     - 从搜索空间中随机选择模型
   * - `PPO Tuner </Tuner/BuiltinTuner.html#PPOTuner>`__
     - PPO Tuner 是基于 PPO 算法的强化学习 Tuner。 `参考论文 <https://arxiv.org/abs/1707.06347>`__


参考 `这里 <ClassicNas.rst>`__ ，了解如何使用经典 NAS 算法。

支持的 One-shot NAS 算法
---------------------------------

NNI 目前支持下面列出的 One-Shot NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 `NNI API <#use-nni-api>`__， 实现其它算法，以使更多人受益。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - 算法简介
   * - `ENAS </NAS/ENAS.html>`__
     - `Efficient Neural Architecture Search via Parameter Sharing <https://arxiv.org/abs/1802.03268>`__. 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 它通过在子模型间共享参数来实现加速和出色的性能指标。
   * - `DARTS </NAS/DARTS.html>`__
     - `DARTS: Differentiable Architecture Search <https://arxiv.org/abs/1806.09055>`__ 介绍了一种用于双级优化的可区分网络体系结构搜索的新算法。
   * - `P-DARTS </NAS/PDARTS.html>`__
     - `Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation <https://arxiv.org/abs/1904.12760>`__ 这篇论文是基于 DARTS 的. 它引入了一种有效的算法，可在搜索过程中逐渐增加搜索的深度。
   * - `SPOS </NAS/SPOS.html>`__
     - 论文 `Single Path One-Shot Neural Architecture Search with Uniform Sampling <https://arxiv.org/abs/1904.00420>`__ 构造了一个采用统一的路径采样方法来训练简化的超网络，并使用进化算法来提高搜索神经网络结构的效率。
   * - `CDARTS </NAS/CDARTS.html>`__
     - `Cyclic Differentiable Architecture Search <https://arxiv.org/abs/****>`__ 在搜索和评估网络之间建立循环反馈机制。 通过引入的循环的可微分架构搜索框架将两个网络集成为一个架构。
   * - `ProxylessNAS </NAS/Proxylessnas.html>`__
     - `ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware <https://arxiv.org/abs/1812.00332>`__. 它删除了代理，直接从大规模目标任务和目标硬件平台进行学习。
   * - `TextNAS </NAS/TextNAS.html>`__
     - `TextNAS: A Neural Architecture Search Space tailored for Text Representation <https://arxiv.org/pdf/1912.10729.pdf>`__. 这是专门用于文本表示的神经网络架构搜索算法。


One-shot 算法 **独立运行，不需要 nnictl**。 NNI 支持 PyTorch 和 TensorFlow 2.x。

这是运行示例的一些常见依赖项。 PyTorch 需要高于 1.2 才能使用 ``BoolTensor``。


* tensorboard
* PyTorch 1.2+
* git

参考  `这里 <NasGuide.rst>`__，了解如何使用 one-shot NAS 算法。

一次性 NAS 可以通过可视化工具来查看。 点 `这里 <./Visualization.rst>`__ 了解更多细节。

搜索空间集合
----------------

NNI 提供了一些预定义的、可被重用的搜索空间。 通过堆叠这些抽取出的单元，用户可以快速复现 NAS 模型。

搜索空间集合包含了以下 NAS 单元：


* `DartsCell <./SearchSpaceZoo.rst#DartsCell>`__
* `ENAS micro <./SearchSpaceZoo.rst#ENASMicroLayer>`__
* `ENAS macro <./SearchSpaceZoo.rst#ENASMacroLayer>`__
* `NAS Bench 201 <./SearchSpaceZoo.rst#nas-bench-201>`__

使用 NNI API 来编写搜索空间
----------------------------------------

在两种场景下需要用于设计和搜索模型的编程接口。


#. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
#. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

为了使用 NNI NAS, 建议用户先通读这篇文档 `the tutorial of NAS API for building search space <./WriteSearchSpace.rst>`__。

NAS 可视化
-----------------

为了帮助用户跟踪指定搜索空间下搜索模型的过程和状态，开发了此可视化工具。 它将搜索空间可视化为超网络，并显示子网络、层和操作的重要性，同时还能显示重要性是如何在搜索过程中变化的。 请参阅 `the document of NAS visualization <./Visualization.rst>`__ 。

参考和反馈
----------------------


* 在Github 中 `提交此功能的 Bug <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__；
* 在Github 中 `提交新功能或请求改进 <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__。
