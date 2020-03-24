# 概述

NNI (Neural Network Intelligence) 是一个工具包，可有效的帮助用户设计并调优机器学习模型的神经网络架构，复杂系统的参数（如超参）等。 NNI has several appealing properties: ease-of-use, scalability, flexibility, and efficiency.

* **Ease-of-use**: NNI can be easily installed through python pip. 只需要在代码中添加几行，就可以利用 NNI 来调优参数。 You can use both the commandline tool and WebUI to work with your experiments.
* **Scalability**: Tuning hyperparameters or the neural architecture often demands a large number of computational resources, while NNI is designed to fully leverage different computation resources, such as remote machines, training platforms (e.g., OpenPAI, Kubernetes). 通过训练平台，可拥有同时运行数百个 Trial 的能力。
* **灵活**：除了内置的算法，NNI 中还可以轻松集成自定义的超参调优算法，神经网络架构搜索算法，提前终止算法等等。 Users can also extend NNI with more training platforms, such as virtual machines, kubernetes service on the cloud. 此外，NNI 还可以连接到外部环境中的特殊应用和模型上。
* **Efficiency**: We are intensively working on more efficient model tuning on both the system and algorithm level. For example, we leverage early feedback to speedup the tuning procedure.

下图显示了 NNI 的体系结构。

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816536-ed055580-2301-11e9-8ad8-605a79ee1b9a.png" alt="绘图" width="700"/>
</p>

## 主要概念

* *Experiment*: One task of, for example, finding out the best hyperparameters of a model, finding out the best neural network architecture, etc. 它由 Trial 和自动机器学习算法所组成。

* *Search Space*: The feasible region for tuning the model. For example, the value range of each hyperparameter.

* *Configuration*: An instance from the search space, that is, each hyperparameter has a specific value.

* *Trial*: An individual attempt at applying a new configuration (e.g., a set of hyperparameter values, a specific neural architecture, etc.). Trial 会基于提供的配置来运行。

* *Tuner*: An AutoML algorithm, which generates a new configuration for the next try. 新的 Trial 会使用这组配置来运行。

* *Assessor*: Analyze a trial's intermediate results (e.g., periodically evaluated accuracy on test dataset) to tell whether this trial can be early stopped or not.

* *Training Platform*: Where trials are executed. 根据 Experiment 的配置，可以是本机，远程服务器组，或其它大规模训练平台（如，OpenPAI，Kubernetes）。

Experiment 的运行过程为：Tuner 接收搜索空间并生成配置。 These configurations will be submitted to training platforms, such as the local machine, remote machines, or training clusters. 执行的性能结果会被返回给 Tuner。 然后，再生成并提交新的配置。

For each experiment, the user only needs to define a search space and update a few lines of code, and then leverage NNI built-in Tuner/Assessor and training platforms to search the best hyperparameters and/or neural architecture. 基本上分为三步：

> 第一步：[定义搜索空间](Tutorial/SearchSpaceSpec.md)
> 
> 第二步：[改动模型代码](TrialExample/Trials.md)
> 
> 第三步：[定义 Experiment 配置](Tutorial/ExperimentConfig.md)

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816627-5d13db80-2302-11e9-8f3e-627e260203d5.jpg" alt="绘图"/>
</p>

For more details about how to run an experiment, please refer to [Get Started](Tutorial/QuickStart.md).

## 核心功能

NNI provides a key capacity to run multiple instances in parallel to find the best combinations of parameters. This feature can be used in various domains, like finding the best hyperparameters for a deep learning model or finding the best configuration for database and other complex systems with real data.

NNI also provides algorithm toolkits for machine learning and deep learning, especially neural architecture search (NAS) algorithms, model compression algorithms, and feature engineering algorithms.

### 超参调优

这是 NNI 最核心、基本的功能，其中提供了许多流行的[自动调优算法](Tuner/BuiltinTuner.md) (即 Tuner) 以及 [提前终止算法](Assessor/BuiltinAssessor.md) (即 Assessor)。 You can follow [Quick Start](Tutorial/QuickStart.md) to tune your model (or system). Basically, there are the above three steps and then starting an NNI experiment.

### 通用 NAS 框架

This NAS framework is for users to easily specify candidate neural architectures, for example, one can specify multiple candidate operations (e.g., separable conv, dilated conv) for a single layer, and specify possible skip connections. NNI 将自动找到最佳候选。 On the other hand, the NAS framework provides a simple interface for another type of user (e.g., NAS algorithm researchers) to implement new NAS algorithms. A detailed description of NAS and its usage can be found [here](NAS/Overview.md).

NNI has support for many one-shot NAS algorithms such as ENAS and DARTS through NNI trial SDK. 使用这些算法时，不需启动 NNI Experiment。 Instead, import an algorithm in your trial code and simply run your trial code. If you want to tune the hyperparameters in the algorithms or want to run multiple instances, you can choose a tuner and start an NNI experiment.

除了 one-shot NAS 外，NAS 还能以 NNI 模式运行，其中每个候选的网络结构都作为独立 Trial 任务运行。 在此模式下，与超参调优类似，必须启动 NNI Experiment 并为 NAS 选择 Tuner。

### 模型压缩

NNI 上的模型压缩包括剪枝和量化算法。 这些算法通过 NNI Trial SDK 提供。 Users can directly use them in their trial code and run the trial code without starting an NNI experiment. A detailed description of model compression and its usage can be found [here](Compressor/Overview.md).

There are different types of hyperparameters in model compression. One type is the hyperparameters in input configuration (e.g., sparsity, quantization bits) to a compression algorithm. The other type is the hyperparameters in compression algorithms. Here, Hyperparameter tuning of NNI can help a lot in finding the best compressed model automatically. 参考[简单示例](Compressor/AutoCompression.md)。

### 自动特征工程

Automatic feature engineering is for users to find the best features for their tasks. A detailed description of automatic feature engineering and its usage can be found [here](FeatureEngineering/Overview.md). 通过 NNI Trial SDK 支持，不必创建 NNI Experiment。 只需在 Trial 代码中加入内置的自动特征工程算法，然后直接运行 Trial 代码。

自动特征工程算法通常有一些超参。 如果要自动调整这些超参，可以利用 NNI 的超参数调优，即选择调优算法（即 Tuner）并启动 NNI Experiment。

## 了解更多信息

* [入门](Tutorial/QuickStart.md)
* [如何为 NNI 调整代码？](TrialExample/Trials.md)
* [NNI 支持哪些 Tuner？](Tuner/BuiltinTuner.md)
* [如何自定义 Tuner？](Tuner/CustomizeTuner.md)
* [NNI 支持哪些 Assessor？](Assessor/BuiltinAssessor.md)
* [如何自定义 Assessor？](Assessor/CustomizeAssessor.md)
* [如何在本机上运行 Experiment？](TrainingService/LocalMode.md)
* [如何在多机上运行 Experiment？](TrainingService/RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](TrainingService/PaiMode.md)
* [示例](TrialExample/MnistExamples.md)
* [NNI 上的神经网络架构搜索](NAS/Overview.md)
* [NNI 上的自动模型压缩](Compressor/Overview.md)
* [NNI 上的自动特征工程](FeatureEngineering/Overview.md)