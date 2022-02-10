.. 1ff18ebada0efec66cd793f1a000f3fe

内置 Tuner
==========

为了让机器学习和深度学习模型适应不同的任务和问题，我们需要进行超参数调优，而自动化调优依赖于优秀的调优算法。NNI 内置了先进的调优算法，并且提供了易于使用的 API。

在 NNI 中，调优算法被称为“tuner”。Tuner 向 trial 发送超参数，接收运行结果从而评估这组超参的性能，然后将下一组超参发送给新的 trial。

下表简要介绍了 NNI 内置的调优算法。点击 tuner 的名称可以查看其安装需求、推荐使用场景、示例配置文件等详细信息。`这篇文章 <../CommunitySharings/HpoComparison.rst>`__ 对比了各个 tuner 在不同场景下的性能。

.. list-table::
  :header-rows: 1
  :widths: auto

  * - Tuner
    - 算法简介

  * - `TPE <./TpeTuner.rst>`__
    - Tree-structured Parzen Estimator (TPE) 是一种基于序列模型的优化方法 (sequential model-based optimization, SMBO)。SMBO方法根据历史数据来顺序地构造模型，从而预估超参性能，并基于此模型来选择新的超参。`参考论文 <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__

  * - `Random Search (随机搜索) <./RandomTuner.rst>`__
    - 随机搜索在超算优化中表现出了令人意外的性能。如果没有对超参分布的先验知识，我们推荐使用随机搜索作为基线方法。`参考论文 <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`__

  * - `Anneal (退火) <./AnnealTuner.rst>`__
    - 朴素退火算法首先基于先验进行采样，然后逐渐逼近实际性能较好的采样点。该算法是随即搜索的变体，利用了反应曲面的平滑性。该实现中退火率不是自适应的。

  * - `Naive Evolution（朴素进化） <./EvolutionTuner.rst>`__
    - 朴素进化算法来自于 Large-Scale Evolution of Image Classifiers。它基于搜索空间随机生成一个种群，在每一代中选择较好的结果，并对其下一代进行变异。朴素进化算法需要很多 Trial 才能取得最优效果，但它也非常简单，易于扩展。`参考论文 <https://arxiv.org/pdf/1703.01041.pdf>`__

  * - `SMAC <./SmacTuner.rst>`__
    - SMAC 是基于序列模型的优化方法 (SMBO)。它利用使用过的最突出的模型（高斯随机过程模型），并将随机森林引入到SMBO中，来处理分类参数。NNI 的 SMAC tuner 封装了 GitHub 上的 `SMAC3 <https://github.com/automl/SMAC3>`__。`参考论文 <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__

      注意：SMAC 算法需要使用 ``pip install nni[SMAC]`` 安装依赖，暂不支持 Windows 操作系统。

  * - `Batch（批处理） <./BatchTuner.rst>`__
    - 批处理允许用户直接提供若干组配置，为每种配置运行一个 trial。

  * - `Grid Search（网格遍历） <./GridsearchTuner.rst>`__
    - 网格遍历会穷举搜索空间中的所有超参组合。

  * - `Hyperband <./HyperbandAdvisor.rst>`__
    - Hyperband 试图用有限的资源探索尽可能多的超参组合。该算法的思路是，首先生成大量超参配置，将每组超参运行较短的一段时间，随后抛弃其中效果较差的一半，让较好的超参继续运行，如此重复多轮。`参考论文 <https://arxiv.org/pdf/1603.06560.pdf>`__

  * - `Metis <./MetisTuner.rst>`__
    - 大多数调参工具仅仅预测最优配置，而 Metis 的优势在于它有两个输出：(a) 最优配置的当前预测结果， 以及 (b) 下一次 trial 的建议。大多数工具假设训练集没有噪声数据，但 Metis 会知道是否需要对某个超参重新采样。`参考论文 <https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/>`__

  * - `BOHB <./BohbAdvisor.rst>`__
    - BOHB 是 Hyperband 算法的后续工作。 Hyperband 在生成新的配置时，没有利用已有的 trial 结果，而本算法利用了 trial 结果。BOHB 中，HB 表示 Hyperband，BO 表示贝叶斯优化（Byesian Optimization）。 BOHB 会建立多个 TPE 模型，从而利用已完成的 Trial 生成新的配置。`参考论文 <https://arxiv.org/abs/1807.01774>`__

  * - `GP (高斯过程) <./GPTuner.rst>`__
    - GP Tuner 是基于序列模型的优化方法 (SMBO)，使用高斯过程进行 surrogate。`参考论文 <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__

  * - `PBT <./PBTTuner.rst>`__
    - PBT Tuner 是一种简单的异步优化算法，在固定的计算资源下，它能有效的联合优化一组模型及其超参来最优化性能。`参考论文 <https://arxiv.org/abs/1711.09846v1>`__

  * - `DNGO <./DngoTuner.rst>`__
    - DNGO 是基于序列模型的优化方法 (SMBO)，该算法使用神经网络（而不是高斯过程）去建模贝叶斯优化中所需要的函数分布。

..  toctree::
    :maxdepth: 1

    TPE <Tuner/TpeTuner>
    Random Search（随机搜索） <Tuner/RandomTuner>
    Anneal（退火） <Tuner/AnnealTuner>
    Naïve Evolution（朴素进化） <Tuner/EvolutionTuner>
    SMAC <Tuner/SmacTuner>
    Metis Tuner <Tuner/MetisTuner>
    Batch Tuner（批处理） <Tuner/BatchTuner>
    Grid Search（网格遍历） <Tuner/GridsearchTuner>
    GP Tuner <Tuner/GPTuner>
    Network Morphism <Tuner/NetworkmorphismTuner>
    Hyperband <Tuner/HyperbandAdvisor>
    BOHB <Tuner/BohbAdvisor>
    PBT Tuner <Tuner/PBTTuner>
    DNGO Tuner <Tuner/DngoTuner>
