# 概述

NNI (Neural Network Intelligence) 是一个工具包，可有效的帮助用户设计并调优机器学习模型的神经网络架构，复杂系统的参数（如超参）等。 NNI 的特性包括：易于使用，可扩展，灵活，高效。

* **易于使用**：NNI 可通过 pip 安装。 只需要在代码中添加几行，就可以利用 NNI 来调优参数。 可使用命令行工具或 Web 界面来查看实验过程。
* **可扩展**：调优超参或网络结构通常需要大量的计算资源。NNI 在设计时就支持了多种不同的计算资源，如远程服务器组，训练平台（如，OpenPAI，Kubernetes），等等。 通过训练平台，可拥有同时运行数百个 Trial 的能力。
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

* *训练平台*：是 Trial 执行的环境。 Depending on your experiment's configuration, it could be your local machine, or remote servers, or large-scale training platform (e.g., PAI, Kubernetes).

Basically, an experiment runs as follows: Tuner receives search space and generates configurations. These configurations will be submitted to training platforms, such as local machine, remote machines, or training clusters. Their performances are reported back to Tuner. Then, new configurations are generated and submitted.

For each experiment, user only needs to define a search space and update a few lines of code, and then leverage NNI built-in Tuner/Assessor and training platforms to search the best hyperparameters and/or neural architecture. There are basically 3 steps:

> 第一步：[定义搜索空间](SearchSpaceSpec.md)
> 
> Step 2: [Update model codes](Trials.md)
> 
> 第三步：[定义 Experiment 配置](ExperimentConfig.md)

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816627-5d13db80-2302-11e9-8f3e-627e260203d5.jpg" alt="drawing"/>
</p>

More details about how to run an experiment, please refer to [Get Started](QuickStart.md).

## Learn More

* [开始使用](QuickStart.md)
* [How to adapt your trial code on NNI?](Trials.md)
* [What are tuners supported by NNI?](Builtin_Tuner.md)
* [How to customize your own tuner?](Customize_Tuner.md)
* [What are assessors supported by NNI?](Builtin_Assessors.md)
* [How to customize your own assessor?](Customize_Assessor.md)
* [How to run an experiment on local?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)
* [Examples](mnist_examples.md)