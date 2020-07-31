# 训练平台

## 什么是训练平台？

NNI 训练平台让用户专注于 AutoML 任务，不需要关心 Trial 实际运行的计算基础架构平台。 当从一个集群迁移到另一个集群时 (如，从本机迁移到 Kubeflow)，用户只需要调整几项配置，能很容易的扩展计算资源。

Users can use training service provided by NNI, to run trial jobs on [local machine](./LocalMode.md), [remote machines](./RemoteMachineMode.md), and on clusters like [PAI](./PaiMode.md), [Kubeflow](./KubeflowMode.md), [FrameworkController](./FrameworkControllerMode.md), [DLTS](./DLTSMode.md) and [AML](./AMLMode.md). 这些都是*内置的训练平台*。

如果需要在计算资源上使用 NNI，可以根据相关接口，轻松构建对其它训练平台的支持。 参考 "[如何实现训练平台](./HowToImplementTrainingService)" 了解详情。

## 如何使用训练平台？

在 Experiment 的 YAML 配置文件中选择并配置好训练平台。 参考相应训练平台的文档来了解如何配置。 同时，[Experiment 文档](../Tutorial/ExperimentConfig)提供了更多详细信息。

然后，需要准备代码目录，将路径填入配置文件的 `codeDir` 字段。 注意，非本机模式下，代码目录会在 Experiment 运行前上传到远程或集群中。 因此，NNI 将文件数量限制到 2000，总大小限制为 300 MB。 如果代码目录中文件太多，可添加 `.nniignore` 文件来排除一部分文件，其用法与 `.gitignore` 类似。 For more details on how to write this file, see [this example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/.nniignore) and the [git documentation](https://git-scm.com/docs/gitignore#_pattern_format).

如果用户需要在 Experiment 使用大文件（如，大规模的数据集），并且不想使用本机模式，可以：1) 在 Trial command 字段中添加命令，每次 Trial 运行前下载数据；或 2) 使用工作节点可访问的共享存储。 通常情况下，训练平台都会有共享存储。 参考每个训练平台的文档，了解详情。

## 内置训练平台

| 训练平台                                                    | 简介                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [__本机__](./LocalMode.md)                                | NNI 支持在本机运行实验，称为 local 模式。 local 模式表示 NNI 会在运行 NNI Manager 进程计算机上运行 Trial 任务，支持 GPU 调度功能。                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [__远程计算机__](./RemoteMachineMode.md)                     | NNI 支持通过 SSH 通道在多台计算机上运行 Experiment，称为 remote 模式。 NNI 需要这些计算机的访问权限，并假定已配置好了深度学习训练环境。 NNI 将在远程计算机上中提交 Trial 任务，并根据 GPU 资源调度 Trial 任务。                                                                                                                                                                                                                                                                                                                                                                                                   |
| [__OpenPAI__](./PaiMode.md)                             | NNI 支持在 [OpenPAI](https://github.com/Microsoft/pai) 上运行 Experiment，即 pai 模式。 在使用 NNI 的 pai 模式前, 需要有 [OpenPAI](https://github.com/Microsoft/pai) 群集的账户。 如果没有 OpenPAI，参考[这里](https://github.com/Microsoft/pai#how-to-deploy)来进行部署。 在 pai 模式中，会在 Docker 创建的容器中运行 Trial 程序。                                                                                                                                                                                                                                                                |
| [__Kubeflow__](./KubeflowMode.md)                       | NNI 支持在 [Kubeflow](https://github.com/kubeflow/kubeflow)上运行，称为 kubeflow 模式。 在开始使用 NNI 的 Kubeflow 模式前，需要有一个 Kubernetes 集群，可以是私有部署的，或者是 [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/zh-cn/services/kubernetes-service/)，并需要一台配置好 [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) 的 Ubuntu 计算机连接到此 Kubernetes 集群。 如果不熟悉 Kubernetes，可先浏览[这里](https://kubernetes.io/docs/tutorials/kubernetes-basics/)。 在 kubeflow 模式下，每个 Trial 程序会在 Kubernetes 集群中作为一个 Kubeflow 作业来运行。 |
| [__FrameworkController__](./FrameworkControllerMode.md) | NNI 支持使用 [FrameworkController](https://github.com/Microsoft/frameworkcontroller)，来运行 Experiment，称之为 frameworkcontroller 模式。 FrameworkController 构建于 Kubernetes 上，用于编排各种应用。这样，可以不用为某个深度学习框架安装 Kubeflow 的 tf-operator 或 pytorch-operator 等。 而直接用 FrameworkController 作为 NNI Experiment 的训练平台。                                                                                                                                                                                                                                            |
| [__DLTS__](./DLTSMode.md)                               | NNI 支持在 [DLTS](https://github.com/microsoft/DLWorkspace.git) 上运行 Experiment，这是一个由微软开源的工具包。                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [__AML__](./AMLMode.md)                                 | NNI supports running an experiment on [AML](https://azure.microsoft.com/en-us/services/machine-learning/) , called aml mode.                                                                                                                                                                                                                                                                                                                                                                                                           |

## 训练平台做了什么？

<p align="center">
<img src="https://user-images.githubusercontent.com/23273522/51816536-ed055580-2301-11e9-8ad8-605a79ee1b9a.png" alt="drawing" width="700"/>
</p>

根据[概述](../Overview)中展示的架构，训练平台会做三件事：1) 启动 Trial; 2) 收集指标，并与 NNI 核心（NNI 管理器）通信；3) 监控 Trial 任务状态。 为了展示训练平台的详细工作原理，下面介绍了训练平台从最开始到第一个 Trial 运行成功的过程。

步骤 1. **验证配置，并准备训练平台。** 训练平台会首先检查用户配置是否正确（例如，身份验证是否有错）。 然后，训练平台会为 Experiment 做准备，创建训练平台可访问的代码目录（`codeDir`）。

```eval_rst
.. Note:: 不同的训练平台会有不同的方法来处理 ``codeDir``。 例如，本机训练平台会直接在 ``codeDir`` 中运行 Trial。 远程训练平台会将 ``codeDir`` 打包成 zip 文件，并上传到每台机器中。 基于 Kubernetes 的训练平台会将 ``codeDir`` 复制到共享存储上，此存储可以由训练平台提供，或者用户在配置文件中指定。
```

步骤 2. **提交第一个 Trial。** 要初始化 Trial，通常（在不重用环境的情况下），NNI 会复制一些文件（包括参数配置，启动脚本等）到训练平台中。 然后，NNI 会通过子进程、SSH、RESTful API 等方式启动 Trial。

```eval_rst
.. Warning:: Trial 当前目录的内容与 ``codeDir`` 会完全一样，但可能是完全不同的路径（甚至不同的计算机）。本机模式是唯一一个所有 Trial 都使用同一个 ``codeDir`` 的训练平台。 其它训练平台，会将步骤 1 中准备好的 ``codeDir``，从共享目录复制到每个 Trial 自己独立的工作目录下。 强烈建议不要依赖于本机模式下的共享行为，这会让 Experiment 很难扩展到其它训练平台上。
```

步骤 3. **收集指标。**  NNI 会监控 Trial 的状态，更新状态（如，从 `WAITING` 到 `RUNNING`，`RUNNING` 到 `SUCCEEDED`，并收集指标。 当前，大部分训练平台都实现为 "主动" 模式，即，训练平台会调用 NNI 管理器上的 RESTful API 来更新指标。 注意，这也需要运行 NNI 管理器的计算机能被工作节点访问到。
