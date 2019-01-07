# 教程 - 尝试不同的调参器和评估器

NNI 提供了简单的方法来设置不同的参数优化算法，以及提前终止策略。NNI 将它们分别称为**调参器**和**评估器**。

**调参器** 指定了为每个尝试生成参数的算法。 在 NNI 中，有两种方法来设置调参器。

1. 直接使用 NNI 提供的调参器
    
        必填字段：builtinTunerName 和 classArgs。 
        

2. 自定义调参器文件
    
        必填字段：codeDirectory, classFileName, className 和 classArgs。
        

### **了解有关调参器的更多信息**

* 有关所需字段的详细定义和用法，参考[配置实验](ExperimentConfig.md)。
* [NNI 最新版本支持的调参器](HowToChooseTuner.md)
* [如何自定义调参器](howto_2_CustomizedTuner.md)

**评估器** 指定了用于提前终止尝试的策略。 在 NNI 中，支持两种方法来设置评估器。

1. 直接使用 NNI 提供的评估器
    
        必填字段：builtinAssessorName 和 classArgs。 
        

2. 自定义评估器文件
    
        必填字段：codeDirectory, classFileName, className 和 classArgs。
        

### **了解有关评估器的更多信息**

* 有关所需字段的详细定义和用法，参考[配置实验](ExperimentConfig.md)。
* 查看[启用评估器](EnableAssessor.md)，了解更多信息。
* [如何自定义评估器](../examples/assessors/README.md)

## **了解更多信息**

* [如何在本机运行实验 (支持多 GPU 卡)？](tutorial_1_CR_exp_local_api.md)
* [如何在多机上运行实验？](tutorial_2_RemoteMachineMode.md)
* [如何在 OpenPAI 上运行实验？](PAIMode.md)