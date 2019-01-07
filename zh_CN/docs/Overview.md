# NNI 概述

NNI (Neural Network Intelligence) 是自动机器学习（AutoML）实验的工具包。 每次实验，用户只需要定义搜索空间，改动几行代码，就能利用 NNI 内置的算法和训练服务来搜索最好的超参组合以及神经网络结构。

> 第一步：[定义搜索空间](SearchSpaceSpec.md)
> 
> 第二步：[改动模型代码](howto_1_WriteTrial.md)
> 
> 第三步：[定义实验配置](ExperimentConfig.md)

<p align="center">
<img src="../../docs/img/3_steps.jpg" alt="drawing"/>
</p>

用户通过命令行工具 [nnictl](../tools/README.md) 创建实验后，守护进程（NNI 管理器）会开始搜索过程。 NNI 管理器不断地通过搜索配置的优化算法来生成参数配置，并通过训练服务组件，在目标训练环境中（例如：本机、远程服务器、云服务等），来调度并运行尝试的任务。 尝试任务的模型精度等结果会返回给优化算法，以便生成更好的参数配置。 NNI 管理器会在找到最佳模型后停止搜索过程。

## 体系结构概述

<p align="center">
<img src="../../docs/img/nni_arch_overview.png" alt="drawing"/>
</p>

User can use the nnictl and/or a visualized Web UI nniboard to monitor and debug a given experiment.

NNI provides a set of examples in the package to get you familiar with the above process. In the following example [/examples/trials/mnist], we had already set up the configuration and updated the training codes for you. You can directly run the following command to start an experiment.

## Key Concepts

**Experiment** in NNI is a method for testing different assumptions (hypotheses) by Trials under conditions constructed and controlled by NNI. During the experiment, one or more conditions are allowed to change in an organized manner and effects of these changes on associated conditions.

### **Trial**

**Trial** in NNI is an individual attempt at applying a set of parameters on a model.

### **Tuner**

**Tuner** in NNI is an implementation of Tuner API for a special tuning algorithm. [Read more about the Tuners supported in the latest NNI release](HowToChooseTuner.md)

### **Assessor**

**Assessor** in NNI is an implementation of Assessor API for optimizing the execution of experiment.

## Learn More

* [Get started](GetStarted.md)
* [Install NNI](Installation.md)
* [Use command line tool nnictl](NNICTLDOC.md)
* [Use NNIBoard](WebUI.md)
* [Use annotation](howto_1_WriteTrial.md#nni-python-annotation)

### **Tutorials**

* [How to run an experiment on local (with multiple GPUs)?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](tutorial_2_RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)