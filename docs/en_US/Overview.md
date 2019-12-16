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

>Step 1: [Define search space](Tutorial/SearchSpaceSpec.md)

>Step 2: [Update model codes](TrialExample/Trials.md)

>Step 3: [Define Experiment](Tutorial/ExperimentConfig.md)


<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816627-5d13db80-2302-11e9-8f3e-627e260203d5.jpg" alt="drawing"/>
</p>

More details about how to run an experiment, please refer to [Get Started](Tutorial/QuickStart.md).

## Core Features

NNI provides a key capacity to run multiple instances in parallel to find best combinations of parameters. This feature can be used in various domains, like find best hyperparameters for a deep learning model, or find best configuration for database and other complex system with real data.

NNI is also like to provide algorithm toolkits for machine learning and deep learning, especially neural architecture search (NAS) algorithms, model compression algorithms, and feature engineering algorithms.

### Hyperparameter Tuning
This is a core and basic feature of NNI, we provide many popular [automatic tuning algorithms](Tuner/BuiltinTuner.md) (i.e., tuner) and [early stop algorithms](Assessor/BuiltinAssessor.md) (i.e., assessor). You could follow [Quick Start](Tutorial/QuickStart.md) to tune your model (or system). Basically, there are the above three steps and then start an NNI experiment.

### General NAS Framework
This NAS framework is for users to easily specify candidate neural architectures, for example, could specify multiple candidate operations (e.g., separable conv, dilated conv) for a single layer, and specify possible skip connections. NNI will find the best candidate automatically. On the other hand, the NAS framework provides simple interface for another type of users (e.g., NAS algorithm researchers) to implement new NAS algorithms. Detailed description and usage can be found [here](NAS/Overview.md).

NNI has supported many one-shot NAS algorithms, such as ENAS, DARTS, through NNI trial SDK. To use these algorithms you do not have to start an NNI experiment. Instead, to import an algorithm in your trial code, and simply run your trial code. If you want to tune the hyperparameters in the algorithms or want to run multiple instances, you could choose a tuner and start an NNI experiment.

Other than one-shot NAS, NAS can also run in a classic mode where each candidate architecture runs as an independent trial job. In this mode, similar to hyperparameter tuning, users have to start an NNI experiment and choose a tuner for NAS.

### Model Compression
Model Compression on NNI includes pruning algorithms and quantization algorithms. These algorithms are provided through NNI trial SDK. Users could directly use them in their trial code and run the trial code without starting an NNI experiment. Detailed description and usage can be found [here](Compressor/Overview.md).

There are different types of hyperparamters in model compression. One type is the hyperparameters in input configuration, e.g., sparsity, quantization bits, to a compression algorithm. The other type is the hyperparamters in compression algorithms. Here, Hyperparameter tuning of NNI could help a lot in finding the best compressed model automatically. A simple example can be found [here](Compressor/AutoCompression.md).

### Automatic Feature Engineering
Automatic feature engineering is for users to find the best features for the following tasks. Detailed description and usage can be found [here](FeatureEngineering/Overview.md). It is supported through NNI trial SDK, which means you do not have to create an NNI experiment. Instead, simply import a built-in auto-feature-engineering algorithm in your trial code and directly run your trial code. 

The auto-feature-engineering algorithms usually have a bunch of hyperparameters themselves. If you want to automatically tune those hyperparameters, you can leverage hyperparameter tuning of NNI, that is, choose a tuning algorithm (i.e., tuner) and start an NNI experiment for it.


## Learn More
* [Get started](Tutorial/QuickStart.md)
* [How to adapt your trial code on NNI?](TrialExample/Trials.md)
* [What are tuners supported by NNI?](Tuner/BuiltinTuner.md)
* [How to customize your own tuner?](Tuner/CustomizeTuner.md)
* [What are assessors supported by NNI?](Assessor/BuiltinAssessor.md)
* [How to customize your own assessor?](Assessor/CustomizeAssessor.md)
* [How to run an experiment on local?](TrainingService/LocalMode.md)
* [How to run an experiment on multiple machines?](TrainingService/RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](TrainingService/PaiMode.md)
* [Examples](TrialExample/MnistExamples.md)
* [Neural Architecture Search on NNI](NAS/Overview.md)
* [Automatic model compression on NNI](Compressor/Overview.md)
* [Automatic feature engineering on NNI](FeatureEngineering/Overview.md)