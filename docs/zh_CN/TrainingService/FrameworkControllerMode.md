# 在 FrameworkController 上运行 Experiment

=== NNI 支持使用 [FrameworkController](https://github.com/Microsoft/frameworkcontroller)，来运行 Experiment，称之为 frameworkcontroller 模式。 FrameworkController 构建于 Kubernetes 上，用于编排各种应用。这样，可以不用为某个深度学习框架安装 Kubeflow 的 tf-operator 或 pytorch-operator 等。 而直接用 FrameworkController 作为 NNI Experiment 的训练平台。

## 私有部署的 Kubernetes 的准备工作

1. 采用 Kubernetes 1.8 或更高版本。 根据[指南](https://kubernetes.io/docs/setup/)来安装 Kubernetes。
2. 配置 **kubeconfig** 文件，NNI 将使用此配置与 Kubernetes API 服务交互。 默认情况下，NNI 管理器会使用 $(HOME)/.kube/config 作为 kubeconfig 文件的路径。 也可以通过环境变量 **KUBECONFIG** 来指定其它 kubeconfig 文件。 根据[指南](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig)了解更多 kubeconfig 的信息。
3. 如果 NNI Trial 作业需要 GPU 资源，需按照[指南](https://github.com/NVIDIA/k8s-device-plugin)来配置 **Kubernetes 下的 Nvidia 插件**。
4. 准备 **NFS 服务器** 并导出通用的装载 (mount)，推荐将 NFS 服务器路径映射到 `root_squash 选项`，否则可能会在 NNI 复制文件到 NFS 时出现权限问题。 参考[页面](https://linux.die.net/man/5/exports)，来了解关于 root_squash 选项，或 **Azure File Storage**。
5. 在安装 NNI 并运行 nnictl 的计算机上安装 **NFS 客户端**。 运行此命令安装 NFSv4 客户端：
  
      ```bash
      apt-get install nfs-common
      ```
      

6. 参考[指南](../Tutorial/QuickStart.md)安装 **NNI**。

## Azure 部署的 Kubernetes 的准备工作

1. NNI 支持基于 Azure Kubernetes Service 的 Kubeflow，参考[指南](https://azure.microsoft.com/zh-cn/services/kubernetes-service/)来设置 Azure Kubernetes Service。
2. 安装 [Azure CLI](https://docs.microsoft.com/zh-cn/cli/azure/install-azure-cli?view=azure-cli-latest) 和 **kubectl**。 使用 `az login` 命令来设置 Azure 账户吗，并将 kubectl 客户端连接到 AKS，参考此[指南](https://docs.microsoft.com/zh-cn/azure/aks/kubernetes-walkthrough#connect-to-the-cluster)。
3. 参考此[指南](https://docs.microsoft.com/zh-cn/azure/storage/common/storage-quickstart-create-account?tabs=portal)来创建 Azure 文件存储账户。 NNI 需要 Azure Storage Service 来存取代码和输出文件。
4. NNI 需要访问密钥来连接 Azure 存储服务，NNI 使用 [Azure Key Vault](https://azure.microsoft.com/zh-cn/services/key-vault/) 服务来保护私钥。 设置 Azure Key Vault 服务，并添加密钥到 Key Vault 中来存取 Azure 存储账户。 参考[指南](https://docs.microsoft.com/zh-cn/azure/key-vault/quick-create-cli)来存储访问密钥。

## 安装 FrameworkController

参考[指南](https://github.com/Microsoft/frameworkcontroller/tree/master/example/run)来在 Kubernetes 集群中配置 FrameworkController。NNI 通过 statefulset 模式来 支持 FrameworkController。 如果集群需要认证，则需要为 FrameworkController 创建服务账户并授予权限，然后将 FrameworkController 服务账户的名称设置到 NNI Experiment 配置中。 [参考文档](https://github.com/Microsoft/frameworkcontroller/tree/master/example/run#run-by-kubernetes-statefulset)

## 设计

参考[Kubeflow 训练平台](KubeflowMode.md)的设计，FrameworkController 训练平台与其类似。

## 示例

FrameworkController 配置文件的格式如下：

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 10h
maxTrialNum: 100
#可选项: local, remote, pai, kubeflow, frameworkcontroller
trainingServicePlatform: frameworkcontroller
searchSpacePath: ~/nni/examples/trials/mnist-tfv1/search_space.json
#可选项: true, false
useAnnotation: false
tuner:
  #可选项: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: maximize
assessor:
  builtinAssessorName: Medianstop
  classArgs:
    optimize_mode: maximize
trial:
  codeDir: ~/nni/examples/trials/mnist-tfv1
  taskRoles:
    - name: worker
      taskNum: 1
      command: python3 mnist.py
      gpuNum: 1
      cpuNum: 1
      memoryMB: 8192
      image: msranni/nni:latest
      frameworkAttemptCompletionPolicy:
        minFailedTaskCount: 1
        minSucceededTaskCount: 1
frameworkcontrollerConfig:
  storage: nfs
  nfs:
    server: {your_nfs_server}
    path: {your_nfs_server_exported_path}
```

如果使用了 Azure Kubernetes Service，需要在 YAML 文件中如下设置 `frameworkcontrollerConfig`：

```yaml
frameworkcontrollerConfig:
  storage: azureStorage
  serviceAccountName: {your_frameworkcontroller_service_account_name}
  keyVault:
    vaultName: {your_vault_name}
    name: {your_secert_name}
  azureStorage:
    accountName: {your_storage_account_name}
    azureShare: {your_azure_share_name}
```

注意：如果用 FrameworkController 模式运行，需要在 YAML 文件中显式设置 `trainingServicePlatform: frameworkcontroller`。

FrameworkController 模式的 Trial 配置格式，是 FrameworkController 官方配置的简化版。参考 [frameworkcontroller 的 tensorflow 示例](https://github.com/Microsoft/frameworkcontroller/blob/master/example/framework/scenario/tensorflow/cpu/tensorflowdistributedtrainingwithcpu.yaml) 了解详情。

frameworkcontroller 模式中的 Trial 配置使用以下主键：

* taskRoles: 配置文件中可以设置多个任务角色，每个任务角色都是在 Kubernetes 集群中的基本执行单元。 
  * name: 任务角色的名字，例如，"worker", "ps", "master"。
  * taskNum: 任务角色的实例数量。
  * command: 在容器中要执行的用户命令。
  * gpuNum: 容器要使用的 GPU 数量。
  * cpuNum: 容器中要使用的 CPU 数量。
  * memoryMB: 容器的内存限制。
  * image: 用来创建 pod，并运行程序的 Docker 映像。
  * frameworkAttemptCompletionPolicy: 运行框架的策略，参考[用户手册](https://github.com/Microsoft/frameworkcontroller/blob/master/doc/user-manual.md#frameworkattemptcompletionpolicy)了解更多信息。 这些策略可以用来控制 pod，例如，如果 worker 任务停止了，但 ps 还在运行，要通过完成策略来停止 ps。

## 如何运行示例

准备好配置文件后，通过运行 nnictl 来启动 Experiment。 在 FrameworkController 上开始 Experiment 的方法与 Kubeflow 类似，可参考[指南](KubeflowMode.md)了解更多信息。

## 版本校验

从 0.6 开始，NNI 支持版本校验，详情参考[这里](PaiMode.md)。