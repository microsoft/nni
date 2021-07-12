##########################
神经网络架构搜索
##########################

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。
最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动调整的模型。
代表工作有 NASNet, ENAS, DARTS, Network Morphism, 以及 Evolution 等。 此外，新的创新不断涌现。

但是，要实现 NAS 算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。
为了促进 NAS 创新 (如, 设计实现新的 NAS 模型，比较不同的 NAS 模型)，
易于使用且灵活的编程接口非常重要。

因此，NNI 设计了 `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__， 它是一个深度学习框架，支持在神经网络模型空间，而不是单个神经网络模型上进行探索性训练。
Retiarii 的探索性训练允许用户以高度灵活的方式表达 *神经网络架构搜索* 和 *超参数调整* 的各种搜索空间。

本文档中的一些常用术语：

* *Model search space（模型搜索空间）* ：它意味着一组模型，用于从中探索/搜索出最佳模型。 有时我们简称为 *search space（搜索空间）* 或 *model space（模型空间）* 。
* *Exploration strategy（探索策略）*：用于探索模型搜索空间的算法。
* *Model evaluator（模型评估器）*：用于训练模型并评估模型的性能。

按照以下说明开始您的 Retiarii 之旅。

..  toctree::
    :maxdepth: 2

    概述 <NAS/Overview>
    快速入门 <NAS/QuickStart>
    构建模型空间 <NAS/construct_space>
    Multi-trial NAS <NAS/multi_trial_nas>
    One-Shot NAS <NAS/one_shot_nas>
    NAS 基准测试 <NAS/Benchmarks>
    NAS API 参考 <NAS/ApiReference>
