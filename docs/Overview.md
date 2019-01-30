# Overview

NNI (Neural Network Intelligence) is a toolkit to help users design and tune machine learning models (e.g., hyperparameters), neural network architectures, or complex system's parameters, in an efficient and automatic way. NNI has several appealing properties: easy-to-use, scalability, flexibility, and efficiency.

* **Easy-to-use**: NNI can be easily installed through python pip. Only several lines need to be added to your code in order to use NNI's power. You can use both commandline tool and WebUI to work with your experiments.
* **Scalability**: Tuning hyperparameters or neural architecture often demands large amount of computation resource, while NNI is designed to fully leverage different computation resources, such as remote machines, training platforms (e.g., OpenPAI, Kubernetes). Hundreds of trials could run in parallel by depending on the capacity of your configured training platforms.
* **Flexibility**: Besides rich built-in algorithms, NNI allows users to customize various hyperparameter tuning algorithms, neural architecture search algorithms, early stopping algorithms, etc. Users could also extend NNI with more training platforms, such as virtual machines, kubernetes service on the cloud. Moreover, NNI can connect to external environments to tune special applications/models on them.
* **Efficiency**: We are intensively working on more efficient model tuning from both system level and algorithm level. For example, leveraging early feedback to speedup tuning procedure.

The figure below shows high-level architecture of NNI.

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816536-ed055580-2301-11e9-8ad8-605a79ee1b9a.png" alt="drawing" width="700"/>
</p> 

## Key Concepts

* *Experiment*: An experiment is one task of, for example, finding out the best hyperparameters of a model, finding out the best neural network architecture. It consists of trials and AutoML algorithms.

* *Search Space*: It means the feasible region for tuning the model. For example, the value range of each hyperparameters.

* *Configuration*: A configuration is an instance from the search space, that is, each hyperparameter has a specific value.

* *Trial*: Trial is an individual attempt at applying a new configuration (e.g., a set of hyperparameter values, a specific nerual architecture). Trial code should be able to run with the provided configuration.

* *Tuner*: Tuner is an AutoML algorithm, which generates a new configuration for the next try. A new trial will run with this configuration.

* *Assessor*: Assessor analyzes trial's intermediate results (e.g., periodically evaluated accuracy on test dataset) to tell whether this trial can be early stopped or not.

* *Training Platform*: It means where trials are executed. Depending on your experiment's configuration, it could be your local machine, or remote servers, or large-scale training platform (e.g., OpenPAI, Kubernetes).

Basically, an experiment runs as follows: Tuner receives search space and generates configurations. These configurations will be submitted to training platforms, such as local machine, remote machines, or training clusters. Their performances are reported back to Tuner. Then, new configurations are generated and submitted.

For each experiment, user only needs to define a search space and update a few lines of code, and then leverage NNI built-in Tuner/Assessor and training platforms to search the best hyperparameters and/or neural architecture. There are basically 3 steps:

>Step 1: [Define search space](SearchSpaceSpec.md)

>Step 2: [Update model codes](Trials.md)

>Step 3: [Define Experiment](ExperimentConfig.md)


<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816627-5d13db80-2302-11e9-8f3e-627e260203d5.jpg" alt="drawing"/>
</p> 

More details about how to run an experiment, please refer to [Get Started](QuickStart.md).

## Learn More
* [Get started](QuickStart.md)
* [How to adapt your trial code on NNI?](Trials.md)
* [What are tuners supported by NNI?](Builtin_Tuner.md)
* [How to customize your own tuner?](Customize_Tuner.md)
* [What are assessors supported by NNI?](Builtin_Assessors.md)
* [How to customize your own assessor?](Customize_Assessor.md)
* [How to run an experiment on local?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)
* [Examples](mnist_examples.md)