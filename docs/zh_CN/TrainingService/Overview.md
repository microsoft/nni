# 训练平台

## 什么是训练平台？

NNI 训练平台让用户专注于 AutoML 任务，不需要关心 Trial 实际运行的计算基础架构平台。 当从一个集群迁移到另一个集群时 (如，从本机迁移到 Kubeflow)，用户只需要调整几项配置，能很容易的扩展计算资源。

NNI 提供的训练平台包括：[本机](./LocalMode.md), [远程计算机](./RemoteMachineMode.md), 以及集群类的 [OpenPAI](./PaiMode.md)，[Kubeflow](./KubeflowMode.md) 和 [FrameworkController](./FrameworkControllerMode.md)。 这些都是*内置的训练平台*。

如果需要在计算资源上使用 NNI，可以根据相关接口，轻松构建对其它训练平台的支持。 参考 "[如何实现训练平台](./HowToImplementTrainingService)" 了解详情。

## 如何使用训练平台？

在 Experiment 的 YAML 配置文件中选择并配置好训练平台。 参考相应训练平台的文档来了解如何配置。 同时，[Experiment 文档](../Tutorial/ExperimentConfig)提供了更多详细信息。

然后，需要准备代码目录，将路径填入配置文件的 `codeDir` 字段。 注意，非本机模式下，代码目录会在 Experiment 运行前上传到远程或集群中。 因此，NNI 将文件数量限制到 2000，总大小限制为 300 MB。 如果代码目录中文件太多，可添加 `.nniignore` 文件来排除一部分文件，其用法与 `.gitignore` 类似。 具体用法可参考 [git 文档](https://git-scm.com/docs/gitignore#_pattern_format)。

如果用户需要在 Experiment 使用大文件（如，大规模的数据集），并且不想使用本机模式，可以：1) 在 Trial command 字段中添加命令，每次 Trial 运行前下载数据；或 2) 使用工作节点可访问的共享存储。 通常情况下，训练平台都会有共享存储。 参考每个训练平台的文档，了解详情。

## 内置训练平台

| 训练平台                                                      | 简介                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [__本机__](./LocalMode.html)                                | NNI 支持在本机运行实验，称为 local 模式。 local 模式表示 NNI 会在运行 NNI Manager 进程计算机上运行 Trial 任务，支持 GPU 调度功能。                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [__远程计算机__](./RemoteMachineMode.html)                     | NNI 支持通过 SSH 通道在多台计算机上运行 Experiment，称为 remote 模式。 NNI 需要这些计算机的访问权限，并假定已配置好了深度学习训练环境。 NNI 将在远程计算机上中提交 Trial 任务，并根据 GPU 资源调度 Trial 任务。                                                                                                                                                                                                                                                                                                                                                                                                   |
| [__OpenPAI__](./PaiMode.html)                             | NNI 支持在 [OpenPAI](https://github.com/Microsoft/pai) 上运行 Experiment，即 pai 模式。 在使用 NNI 的 pai 模式前, 需要有 [OpenPAI](https://github.com/Microsoft/pai) 群集的账户。 如果没有 OpenPAI，参考[这里](https://github.com/Microsoft/pai#how-to-deploy)来进行部署。 在 pai 模式中，会在 Docker 创建的容器中运行 Trial 程序。                                                                                                                                                                                                                                                                |
| [__Kubeflow__](./KubeflowMode.html)                       | NNI 支持在 [Kubeflow](https://github.com/kubeflow/kubeflow)上运行，称为 kubeflow 模式。 在开始使用 NNI 的 Kubeflow 模式前，需要有一个 Kubernetes 集群，可以是私有部署的，或者是 [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/zh-cn/services/kubernetes-service/)，并需要一台配置好 [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) 的 Ubuntu 计算机连接到此 Kubernetes 集群。 如果不熟悉 Kubernetes，可先浏览[这里](https://kubernetes.io/docs/tutorials/kubernetes-basics/)。 在 kubeflow 模式下，每个 Trial 程序会在 Kubernetes 集群中作为一个 Kubeflow 作业来运行。 |
| [__FrameworkController__](./FrameworkControllerMode.html) | NNI 支持使用 [FrameworkController](https://github.com/Microsoft/frameworkcontroller)，来运行 Experiment，称之为 frameworkcontroller 模式。 FrameworkController 构建于 Kubernetes 上，用于编排各种应用。这样，可以不用为某个深度学习框架安装 Kubeflow 的 tf-operator 或 pytorch-operator 等。 而直接用 FrameworkController 作为 NNI Experiment 的训练平台。                                                                                                                                                                                                                                            |
| [__DLTS__](./DLTSMode.html)                               | NNI supports running experiment using [DLTS](https://github.com/microsoft/DLWorkspace.git), which is an open source toolkit, developed by Microsoft, that allows AI scientists to spin up an AI cluster in turn-key fashion.                                                                                                                                                                                                                                                                                                           |

## What does Training Service do?

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816536-ed055580-2301-11e9-8ad8-605a79ee1b9a.png" alt="drawing" width="700"/>
</p>

According to the architecture shown in [Overview](../Overview), training service (platform) is actually responsible for two events: 1) initiating a new trial; 2) collecting metrics and communicating with NNI core (NNI manager); 3) monitoring trial job status. To demonstrated in detail how training service works, we show the workflow of training service from the very beginning to the moment when first trial succeeds.

Step 1. **Validate config and prepare the training platform.** Training service will first check whether the training platform user specifies is valid (e.g., is there anything wrong with authentication). After that, training service will start to prepare for the experiment by making the code directory (`codeDir`) accessible to training platform.

```eval_rst
.. Note:: Different training services have different ways to handle ``codeDir``. For example, local training service directly runs trials in ``codeDir``. Remote training service packs ``codeDir`` into a zip and uploads it to each machine. K8S-based training services copy ``codeDir`` onto a shared storage, which is either provided by training platform itself, or configured by users in config file.
```

Step 2. **Submit the first trial.** To initiate a trial, usually (in non-reuse mode), NNI copies another few files (including parameters, launch script and etc.) onto training platform. After that, NNI launches the trial through subprocess, SSH, RESTful API, and etc.

```eval_rst
.. Warning:: The working directory of trial command has exactly the same content as ``codeDir``, but can have a differen path (even on differen machines) Local mode is the only training service that shares one ``codeDir`` across all trials. Other training services copies a ``codeDir`` from the shared copy prepared in step 1 and each trial has an independent working directory. We strongly advise users not to rely on the shared behavior in local mode, as it will make your experiments difficult to scale to other training services.
```

Step 3. **Collect metrics.**  NNI then monitors the status of trial, updates the status (e.g., from `WAITING` to `RUNNING`, `RUNNING` to `SUCCEEDED`) recorded, and also collects the metrics. Currently, most training services are implemented in an "active" way, i.e., training service will call the RESTful API on NNI manager to update the metrics. Note that this usually requires the machine that runs NNI manager to be at least accessible to the worker node.
