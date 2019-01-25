# Tutorial - Try different Tuners and Assessors

NNI provides an easy to adopt approach to set up parameter tuning algorithms as well as early stop policies, we call them **Tuners** and **Assessors**.
  
**Tuner** specifies the algorithm you use to generate hyperparameter sets for each trial. In NNI, we support two approaches to set the tuner. 
1. Directly use tuner provided by nni sdk

        required fields: builtinTunerName and classArgs. 

2. Customize your own tuner file

        required fields: codeDirectory, classFileName, className and classArgs.

### **Learn More about tuners**
* For detailed defintion and usage about the required field, please refer to [Config an experiment](ExperimentConfig.md)
* [Tuners in the latest NNI release](HowToChooseTuner.md)
* [How to implement your own tuner](howto_2_CustomizedTuner.md)


**Assessor** specifies the algorithm you use to apply early stop policy. In NNI, there are two approaches to set the assessor.
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
