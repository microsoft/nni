# 教程 - 使用不同的 Tuner 和 Assessor

NNI 提供了简单的方法来设置不同的参数优化算法，以及提前终止策略。NNI 将它们分别称为**Tuner（调参器）**和**Assessor（评估器）**。

**Tuner（调参器）** 指定了为每个 Trial 生成参数的算法。 在 NNI 中，有两种方法来设置 Tuner。

1. 直接使用 NNI 提供的 Tuner
    
        必填字段：builtinTunerName 和 classArgs。 
        

2. 自定义 Tuner
    
        必填字段：codeDirectory, classFileName, className 和 classArgs。
        

### **了解有关 Tuner 的更多信息**

* 有关所需字段的详细定义和用法，参考[配置 Experiment](ExperimentConfig.md)。
* [NNI 最新版本支持的 Tuner](HowToChooseTuner.md)
* [如何自定义 Tuner](howto_2_CustomizedTuner.md)

**Assessor（评估器）** 指定了用于提前终止 Trial 的策略。 在 NNI 中，支持两种方法来设置 Assessor。

1. 直接使用 NNI 提供的 Assessor
    
        必填字段：builtinAssessorName 和 classArgs。 
        

2. 自定义 Assessor 代码
    
        必填字段：codeDirectory, classFileName, className 和 classArgs。
        

### **了解 Assessor 的更多信息**

* 有关所需字段的详细定义和用法，参考[配置 Experiment](ExperimentConfig.md)。
* 查看[启用 Assessor](EnableAssessor.md)，了解更多信息。
* [如何自定义 Assessor](../examples/assessors/README.md)

## **了解更多信息**

* [如何在本机运行 Experiment (支持多 GPU 卡)？](tutorial_1_CR_exp_local_api.md)
* [如何在多机上运行 Experiment？](tutorial_2_RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](PAIMode.md)