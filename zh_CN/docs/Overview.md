# NNI 概述

NNI (Neural Network Intelligence) 是自动机器学习（AutoML）Experiment 的工具包。 每次 Experiment，用户只需要定义搜索空间，改动几行代码，就能利用 NNI 内置的算法和训练服务来搜索最好的超参组合以及神经网络结构。

> 第一步：[定义搜索空间](SearchSpaceSpec.md)
> 
> 第二步：[改动模型代码](howto_1_WriteTrial.md)
> 
> 第三步：[定义 Experiment 配置](ExperimentConfig.md)

<p align="center">
<img src="./img/3_steps.jpg" alt="drawing"/>
</p>

用户通过命令行工具 [nnictl](../tools/README.md) 创建 Experiment 后，守护进程（NNI 管理器）会开始搜索过程。 NNI 管理器不断地通过搜索配置的优化算法来生成参数配置，并通过训练服务组件，在目标训练环境中（例如：本机、远程服务器、云服务等），来调度并运行 Trial 的任务。 Trial 任务的模型精度等结果会返回给优化算法，以便生成更好的参数配置。 NNI 管理器会在找到最佳模型后停止搜索过程。

## 体系结构概述

<p align="center">
<img src="./img/nni_arch_overview.png" alt="drawing"/>
</p>

用户可以用 nnictl 或可视化的 WEB 界面 NNIBoard 来查看并调试指定的 Experiment。

NNI 提供了一组样例来帮助熟悉以上过程。

## 主要概念

**Experiment（实验）**，在 NNI 中是通过 Trial（尝试）在给定的条件来测试不同的假设情况。 在 Experiment 过程中，会有条理的修改一个或多个条件，以便测试它们对相关条件的影响。

### **Trial（尝试）**

**Trial（尝试）**是将一组参数在模型上独立的一次尝试。

### **Tuner（调参器）**

**Tuner（调参器）**，在 NNI 中是实现了 Tuner API 的某个超参调优算法。 [了解 NNI 中最新内置的 Tuner](HowToChooseTuner.md)

### **Assessor（评估器）**

**Assessor（评估器）**，实现了 Assessor API，用来加速 Experiment 执行过程。

## 了解更多信息

* [开始使用](GetStarted.md)
* [安装 NNI](Installation.md)
* [使用命令行工具 nnictl](NNICTLDOC.md)
* [使用 NNIBoard](WebUI.md)
* [使用标记](howto_1_WriteTrial.md#nni-python-annotation)

### **教程**

* [如何在本机运行 Experiment (支持多 GPU 卡)？](tutorial_1_CR_exp_local_api.md)
* [如何在多机上运行 Experiment？](tutorial_2_RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](PAIMode.md)