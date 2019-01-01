# NNI Overview

NNI (Neural Network Intelligence) is a toolkit to help users design and tune machine learning models (e.g., hyperparameters), neural network architectures, or complex system's parameters, in an efficient and automatic way. NNI has several appealing properties: easy-to-use, scalability, flexibility, extensibility, efficiency.

* **Easy-to-use**: NNI can be easily installed through python pip. There are only several lines to added to your code in order to use NNI's power to tune your model. And you can use both a commandline tool and WebUI to view and control your experiments. 
* **Scalability**: Tuning hyperparameters or neural architecture often demands large amount of computation resource, while NNI is designed to fully leverage different computation resources, such as remote machines, training platforms (e.g., PAI, Kubernetes). NNI could run thousands of parallel trials depends on the capacity of your configured training platforms.
* **Flexibility**: Besides rich built-in algorithms, NNI allows users to customize various hyperparameter tuning algorithms, neural architecture search algorithms, early stopping algorithms, etc.
* **Extensibility**: Users could extend NNI with more training platforms, running trials on those computation resources. For example, connecting NNI with virtual machines, kubernetes service, etc. on the cloud, such as Azure, AWS, Aliyun. Users could also connect NNI to external environments to tune the applications/models on them.
* **Efficiency**: We are intensively working on more efficient model tuning from both system level and algorithm level. For example, leveraging early feedback to speedup tuning procedure.

## Key Concepts

* *Experiment*: An experiment is one task of, for example, finding out the best hyperparameters of a model, finding out the best neural network architecture. It consists of trials and AutoML algorithms.

* *Search Space*: It means the feasible region for tuning the model. For example, the value range of each hyperparameters.

* *Configuration*: A configuration is an instance from the search space, that is, each hyperparameter has a specific value.

* *Trial*: Trial is an individual attempt at applying a new configuration, for example, a set of hyperparameter values on a model, or a specific nerual architecture.

* *Tuner*: Tuner is an AutoML algorithm, which generates a new configuration for the next try. A new trial runs with this configuration.

* *Assessor*: Assessor analyzes trial's intermediate results (e.g., periodically evaluated accuracy on test dataset) to tell whether this trial can be early stopped or not.

Basically, an experiment runs as follows: Tuner receives search space and generates configurations. These configurations will be submitted to training platforms, such as local machine, remote machines, or training clusters. Their performances are reported back to Tuner. Then, new configurations are generated and submitted.

For each experiment, user only needs to define a search space and update a few lines of code, and then leverage NNI built-in Tuner/Assessor and training platforms to search the best hyper parameters and/or neural architecture. There are basically 3 steps:

>Step 1: [Define search space](SearchSpaceSpec.md)

>Step 2: [Update model codes](howto_1_WriteTrial.md)

>Step 3: [Define Experiment](ExperimentConfig.md)


<p align="center">
<img src="./img/3_steps.jpg" alt="drawing"/>
</p> 

More details about how to run an experiment, please refer to [Get Started]().

## Learn More
* [Get started](GetStarted.md)
* [How to adapt your trial code on NNI?]()
* [What are tuners supported by NNI?]()
* [How to customize your own tuner?]()
* [What are assessors supported by NNI?]()
* [How to customize your own assessor?]()
* [How to run an experiment on local?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](tutorial_2_RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)
* [How to do trouble shooting when using NNI?]()
* [Examples]()
* [Reference]()