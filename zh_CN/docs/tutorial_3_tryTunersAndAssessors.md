# 教程 - 尝试不同的调参器和评估器

NNI 提供了简单的方法来设置不同的参数优化算法，以及提前终止策略。NNI 将它们分别称为**调参器**和**评估器**。

**调参器** 指定了为每个尝试生成参数的算法。 在 NNI 中，有两种方法来设置调参器。

1. 直接使用 NNI 提供的调参器
    
        必填字段：builtinTunerName 和 classArgs。 
        

2. Customize your own tuner file
    
        required fields: codeDirectory, classFileName, className and classArgs.
        

### **Learn More about tuners**

* For detailed defintion and usage aobut the required field, please refer to [Config an experiment](ExperimentConfig.md)
* [Tuners in the latest NNI release](HowToChooseTuner.md)
* [How to implement your own tuner](howto_2_CustomizedTuner.md)

**Assessor** specifies the algorithm you use to apply early stop policy. In NNI, there are two approaches to set theassessor.

1. Directly use assessor provided by nni sdk
    
        required fields: builtinAssessorName and classArgs. 
        

2. Customize your own assessor file
    
        required fields: codeDirectory, classFileName, className and classArgs.
        

### **Learn More about assessor**

* For detailed defintion and usage aobut the required field, please refer to [Config an experiment](ExperimentConfig.md)
* Find more about the detailed instruction about [enable assessor](EnableAssessor.md)
* [How to implement your own assessor](../examples/assessors/README.md)

## **Learn More**

* [How to run an experiment on local (with multiple GPUs)?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](tutorial_2_RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)