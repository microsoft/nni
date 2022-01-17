Overview
========

NNI (Neural Network Intelligence) is a toolkit to help users design and tune machine learning models (e.g., hyperparameters), neural network architectures, or complex system's parameters, in an efficient and automatic way. NNI has several appealing properties: ease-of-use, scalability, flexibility, and efficiency.


* **Ease-of-use**\ : NNI can be easily installed through python pip. Only several lines need to be added to your code in order to use NNI's power. You can use both the commandline tool and WebUI to work with your experiments.
* **Scalability**\ : Tuning hyperparameters or the neural architecture often demands a large number of computational resources, while NNI is designed to fully leverage different computation resources, such as remote machines, training platforms (e.g., OpenPAI, Kubernetes). Hundreds of trials could run in parallel by depending on the capacity of your configured training platforms.
* **Flexibility**\ : Besides rich built-in algorithms, NNI allows users to customize various hyperparameter tuning algorithms, neural architecture search algorithms, early stopping algorithms, etc. Users can also extend NNI with more training platforms, such as virtual machines, kubernetes service on the cloud. Moreover, NNI can connect to external environments to tune special applications/models on them.
* **Efficiency**\ : We are intensively working on more efficient model tuning on both the system and algorithm level. For example, we leverage early feedback to speedup the tuning procedure.

The figure below shows high-level architecture of NNI.


.. raw:: html

   <p align="center">
   <img src="https://user-images.githubusercontent.com/16907603/92089316-94147200-ee00-11ea-9944-bf3c4544257f.png" alt="drawing" width="700"/>
   </p>

