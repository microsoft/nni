# Run an Experiment on FrameworkController

=== NNI supports running experiment using [FrameworkController](https://github.com/Microsoft/frameworkcontroller), called frameworkcontroller mode. FrameworkController is built to orchestrate all kinds of applications on Kubernetes, you don't need to install Kubeflow for specific deep learning framework like tf-operator or pytorch-operator. Now you can use FrameworkController as the training service to run NNI experiment.

## 私有部署的 Kubernetes 的准备工作

1. 采用 Kubernetes 1.8 或更高版本。 根据[指南](https://kubernetes.io/docs/setup/)来安装 Kubernetes。
2. Prepare a **kubeconfig** file, which will be used by NNI to interact with your Kubernetes API server. 默认情况下，NNI 管理器会使用 $(HOME)/.kube/config 作为 kubeconfig 文件的路径。 也可以通过环境变量 **KUBECONFIG** 来指定其它 kubeconfig 文件。 根据[指南](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig)了解更多 kubeconfig 的信息。
3. 如果 NNI Trial 作业需要 GPU 资源，需按照[指南](https://github.com/NVIDIA/k8s-device-plugin)来配置 **Kubernetes 下的 Nvidia 插件**。
4. 准备 **NFS 服务器** 并导出通用的装载 (mount)，推荐将 NFS 服务器路径映射到 `root_squash 选项`，否则可能会在 NNI 复制文件到 NFS 时出现权限问题。 参考[页面](https://linux.die.net/man/5/exports)，来了解关于 root_squash 选项，或 **Azure File Storage**。
5. Install **NFS client** on the machine where you install NNI and run nnictl to create experiment. Run this command to install NFSv4 client:
  
      ```bash
      apt-get install nfs-common
      ```
      

6. 参考[指南](QuickStart.md)安装 **NNI**。

## Azure 部署的 Kubernetes 的准备工作

1. NNI support Kubeflow based on Azure Kubernetes Service, follow the [guideline](https://azure.microsoft.com/en-us/services/kubernetes-service/) to set up Azure Kubernetes Service.
2. 安装 [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) 和 **kubectl**。 使用 `az login` 命令来设置 Azure 账户吗，并将 kubectl 客户端连接到 AKS，参考此[指南](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster)。
3. 参考此[指南](https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal)来创建 Azure 文件存储账户。 NNI 需要 Azure Storage Service 来存取代码和输出文件。
4. NNI 需要访问密钥来连接 Azure 存储服务，NNI 使用 [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/) 服务来保护私钥。 设置 Azure Key Vault 服务，并添加密钥到 Key Vault 中来存取 Azure 存储账户。 参考[指南](https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli)来存储访问密钥。

## Setup FrameworkController

Follow the [guideline](https://github.com/Microsoft/frameworkcontroller/tree/master/example/run) to set up FrameworkController in the Kubernetes cluster, NNI supports FrameworkController by the stateful set mode.

## 设计

Please refer the design of [Kubeflow training service](./KubeflowMode.md), FrameworkController training service pipeline is similar.

## 样例

The FrameworkController config file format is:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 10h
maxTrialNum: 100
#choice: local, remote, pai, kubeflow, frameworkcontroller
trainingServicePlatform: frameworkcontroller
searchSpacePath: ~/nni/examples/trials/mnist/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
assessor:
  builtinAssessorName: Medianstop
  classArgs:
    optimize_mode: maximize
  gpuNum: 0
trial:
  codeDir: ~/nni/examples/trials/mnist
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
  keyVault:
    vaultName: {your_vault_name}
    name: {your_secert_name}
  azureStorage:
    accountName: {your_storage_account_name}
    azureShare: {your_azure_share_name}
```

注意：如果用 FrameworkController 模式运行，需要在 YAML 文件中显式设置 `trainingServicePlatform: frameworkcontroller`。

The trial's config format for NNI frameworkcontroller mode is a simple version of FrameworkController's official config, you could refer the [Tensorflow example of FrameworkController](https://github.com/Microsoft/frameworkcontroller/blob/master/example/framework/scenario/tensorflow/cpu/tensorflowdistributedtrainingwithcpu.yaml) for deep understanding.

Trial configuration in frameworkcontroller mode have the following configuration keys:

* taskRoles: you could set multiple task roles in config file, and each task role is a basic unit to process in Kubernetes cluster. 
  * name: 任务角色的名字，例如，"worker", "ps", "master"。
  * taskNum: 任务角色的实例数量。
  * command: 在容器中要执行的用户命令。
  * gpuNum: 容器要使用的 GPU 数量。
  * cpuNum: 容器中要使用的 CPU 数量。
  * memoryMB: 容器的内存限制。
  * image: 用来创建 pod，并运行程序的 Docker 映像。
  * frameworkAttemptCompletionPolicy: 运行框架的策略，参考[用户手册](https://github.com/Microsoft/frameworkcontroller/blob/master/doc/user-manual.md#frameworkattemptcompletionpolicy)了解更多信息。 Users could use the policy to control the pod, for example, if ps does not stop, only worker stops, The completion policy could helps stop ps.

## 如何运行示例

After you prepare a config file, you could run your experiment by nnictl. The way to start an experiment on FrameworkController is similar to Kubeflow, please refer the [document](./KubeflowMode.md) for more information.

## 版本校验

NNI support version check feature in since version 0.6, [refer](PaiMode.md)