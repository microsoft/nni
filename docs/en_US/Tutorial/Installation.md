# Plan your topology
For different computation needs, NNI supports various out-of-the-box training modes, including: local machine, remote servers, pai, kubeflow and frameworklauncher. User can also customize their own training service in case of need.

Before starting the installation, you should firstly plan for the topology. For new users, we would always recommend you to start with `local` mode, as it is the simplest topology, all the scripts and computation are happen on one single machine. You only need to install NNI on the machine you will run the training and execute the nni cmdline on that machine. 

But in most cases, you might need more powerful computes on remote servers or the cloud. In the case of using remote servers, you will need to consider `remote` mode or the other training modes NNI supports. As illustrated below, to run trainings on remote servers, you will need to set up a **NNI Manager Node** and a set of **NNI Training Nodes**. The set up are all through the following `pip install` command in the following section.

Note: For Windows, NNI only supports it work as **NNI Manager Node**. Anyone who has interest on support windows as training nodes as well could comment or even provide contributes to the following issue: Support Windows as Remote Training Node [#1973](https://github.com/microsoft/nni/issues/1973).


![image](https://user-images.githubusercontent.com/39592018/72716629-82ea5e00-3bad-11ea-8536-b7a21fa22417.png)

# Installation

We support installation on Linux, Mac and Windows.

## **Installation on Linux & Mac**

* __Install NNI through pip__

  Prerequisite: `python >= 3.5`

  ```bash
  python3 -m pip install --upgrade nni
  ```

* __Install NNI through source code__

  Prerequisite: `python >=3.5`, `git`, `wget`

  ```bash
  git clone -b v0.8 https://github.com/Microsoft/nni.git
  cd nni
  ./install.sh
  ```

* __Install NNI in docker image__

  You can also install NNI in a docker image. Please follow the instructions [here](https://github.com/Microsoft/nni/tree/master/deployment/docker/README.md) to build NNI docker image. The NNI docker image can also be retrieved from Docker Hub through the command `docker pull msranni/nni:latest`.

## **Installation on Windows**

  Anaconda or Miniconda is highly recommended.

* __Install NNI through pip__

  Prerequisite: `python(64-bit) >= 3.5`

  ```bash
  python -m pip install --upgrade nni
  ```

* __Install NNI through source code__

  Prerequisite: `python >=3.5`, `git`, `PowerShell`.

  ```bash
  git clone -b v0.8 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
  ```

## **System requirements**

Below are the minimum system requirements for NNI on Linux. Due to potential programming changes, the minimum system requirements for NNI may change over time.

||Minimum Requirements|Recommended Specifications|
|---|---|---|
|**Operating System**|Ubuntu 16.04 or above|Ubuntu 16.04 or above|
|**CPU**|Intel® Core™ i3 or AMD Phenom™ X3 8650|Intel® Core™ i5 or AMD Phenom™ II X3 or better|
|**GPU**|NVIDIA® GeForce® GTX 460|NVIDIA® GeForce® GTX 660 or better|
|**Memory**|4 GB RAM|6 GB RAM|
|**Storage**|30 GB available hare drive space|
|**Internet**|Boardband internet connection|
|**Resolution**|1024 x 768 minimum display resolution|

Below are the minimum system requirements for NNI on macOS. Due to potential programming changes, the minimum system requirements for NNI may change over time.

||Minimum Requirements|Recommended Specifications|
|---|---|---|
|**Operating System**|macOS 10.14.1 (latest version)|macOS 10.14.1 (latest version)|
|**CPU**|Intel® Core™ i5-760 or better|Intel® Core™ i7-4770 or better|
|**GPU**|NVIDIA® GeForce® GT 750M or AMD Radeon™ R9 M290 or better|AMD Radeon™ R9 M395X or better|
|**Memory**|4 GB RAM|8 GB RAM|
|**Storage**|70GB available space 7200 RPM HDD|70GB available space SSD|
|**Internet**|Boardband internet connection|
|**Resolution**|1024 x 768 minimum display resolution|

Below are the minimum system requirements for NNI on Windows, Windows 10.1809 is well tested and recommend. Due to potential programming changes, the minimum system requirements for NNI may change over time.

||Minimum Requirements|Recommended Specifications|
|---|---|---|
|**Operating System**|Windows 10|Windows 10|
|**CPU**|Intel® Core™ i3 or AMD Phenom™ X3 8650|Intel® Core™ i5 or AMD Phenom™ II X3 or better|
|**GPU**|NVIDIA® GeForce® GTX 460|NVIDIA® GeForce® GTX 660 or better|
|**Memory**|4 GB RAM|6 GB RAM|
|**Storage**|30 GB available hare drive space|
|**Internet**|Boardband internet connection|
|**Resolution**|1024 x 768 minimum display resolution|

## Further reading

* [Overview](../Overview.md)
* [Use command line tool nnictl](Nnictl.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](../TrainingService/LocalMode.md)
* [How to run an experiment on multiple machines?](../TrainingService/RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](../TrainingService/PaiMode.md)
* [How to run an experiment on Kubernetes through Kubeflow?](../TrainingService/KubeflowMode.md)
* [How to run an experiment on Kubernetes through FrameworkController?](../TrainingService/FrameworkControllerMode.md)
