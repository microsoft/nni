# 概述

NNI (Neural Network Intelligence) 是一个工具包，可有效的帮助用户设计并调优机器学习模型的神经网络架构，复杂系统的参数（如超参）等。 NNI 的特性包括：易于使用，可扩展，灵活，高效。

* **易于使用**：NNI 可通过 pip 安装。 只需要在代码中添加几行，就可以利用 NNI 来调优参数。 可使用命令行工具或 Web 界面来查看实验过程。
* **可扩展**：调优超参或网络结构通常需要大量的计算资源。NNI 在设计时就支持了多种不同的计算资源，如远程服务器组，训练平台（如：OpenPAI，Kubernetes），等等。 通过训练平台，可拥有同时运行数百个 Trial 的能力。
* **灵活**：除了内置的算法，NNI 中还可以轻松集成自定义的超参调优算法，神经网络架构搜索算法，提前终止算法等等。 还可以将 NNI 连接到更多的训练平台上，如云中的虚拟机集群，Kubernetes 服务等等。 此外，NNI 还可以连接到外部环境中的特殊应用和模型上。
* **高效**：NNI 在系统及算法级别上不停的优化。 例如：通过 Trial 早期的反馈来加速调优过程。

下图显示了 NNI 的体系结构。

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816536-ed055580-2301-11e9-8ad8-605a79ee1b9a.png" alt="drawing" width="700"/>
</p>

## 主要概念

* *Experiment（实验）*：实验是一次找到模型的最佳超参组合，或最好的神经网络架构的任务。 它由 Trial 和自动机器学习算法所组成。

* *搜索空间*：是模型调优的范围。 例如，超参的取值范围。

* *Configuration（配置）*：配置是来自搜索空间的一个参数实例，每个超参都会有一个特定的值。

* *Trial*: Trial 是一次尝试，它会使用某组配置（例如，一组超参值，或者特定的神经网络架构）。 Trial 会基于提供的配置来运行。

* *Tuner*: Tuner 是一个自动机器学习算法，会为下一个 Trial 生成新的配置。 新的 Trial 会使用这组配置来运行。

* *Assessor*：Assessor 分析 Trial 的中间结果（例如，测试数据集上定期的精度），来确定 Trial 是否应该被提前终止。

* *训练平台*：是 Trial 的执行环境。 根据 Experiment 的配置，可以是本机，远程服务器组，或其它大规模训练平台（如，OpenPAI，Kubernetes）。

Experiment 的运行过程为：Tuner 接收搜索空间并生成配置。 这些配置将被提交到训练平台，如本机，远程服务器组或训练集群。 执行的性能结果会被返回给 Tuner。 然后，再生成并提交新的配置。

每次 Experiment 执行时，用户只需要定义搜索空间，改动几行代码，就能利用 NNI 内置的 Tuner/Assessor 和训练服务来搜索最好的超参组合以及神经网络结构。 基本上分为三步：

> 第一步：[定义搜索空间](SearchSpaceSpec.md)
> 
> 第二步：[改动模型代码](Trials.md)
> 
> 第三步：[定义 Experiment 配置](ExperimentConfig.md)

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816627-5d13db80-2302-11e9-8f3e-627e260203d5.jpg" alt="drawing"/>
</p>

更多 Experiment 运行的详情，参考[快速入门](QuickStart.md)。

## 了解更多信息

* [开始使用](QuickStart.md)
* [如何为 NNI 调整代码？](Trials.md)
* [NNI 支持哪些 Tuner？](Builtin_Tuner.md)
* [如何自定义 Tuner？](Customize_Tuner.md)
* [NNI 支持哪些 Assessor？](Builtin_Assessors.md)
* [如何自定义 Assessor？](Customize_Assessor.md)
* [如何在本机上运行 Experiment？](tutorial_1_CR_exp_local_api.md)
* [如何在多机上运行 Experiment？](RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](PAIMode.md)
* [样例](mnist_examples.md)