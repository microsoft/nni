# **在 FrameworkController 上运行实验**

NNI 支持使用 [FrameworkController](https://github.com/Microsoft/frameworkcontroller)，来运行实验，称之为 frameworkcontroller 模式。 FrameworkController 构建于 Kubernetes 上，用于编排各种应用。这样，可以不用为某个深度学习框架安装 Kubeflow 的 tf-operator 的 pytorch-operator 等。 而直接用 frameworkcontroller 作为 NNI 实验的训练服务。

## 私有部署的 Kubernetes 的准备工作

1. 采用 Kubernetes 1.8 或更高版本。 根据[指南](https://kubernetes.io/docs/setup/)来安装 Kubernetes。
2. 配置 **kubeconfig** 文件，NNI 将使用此配置与 Kubernetes API 服务交互。 默认情况下，NNI 管理器会使用 $(HOME)/.kube/config 作为 kubeconfig 文件的路径。 也可以通过环境变量 **KUBECONFIG** 来指定其它 kubeconfig 文件。 根据[指南](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig)了解更多 kubeconfig 的信息。 
3. 如果 NNI 尝试作业需要 GPU 资源，需按照[指南](https://github.com/NVIDIA/k8s-device-plugin)来配置 **Kubernetes 下的 Nvidia 插件**。
4. 准备 **NFS 服务器** 并导出通用的装载 (mount)，推荐将 NFS 服务器路径映射到 `root_squash 选项`，否则可能会在 NNI 复制文件到 NFS 时出现权限问题。 参考[页面](https://linux.die.net/man/5/exports)，来了解关于 root_squash 选项，或 **Azure File Storage**。 
5. 在安装 NNI 并运行 nnictl 的计算机上安装 **NFS 客户端**。 运行此命令安装 NFSv4 客户端： ```apt-get install nfs-common```

6. 参考[指南](GetStarted.md)安装 **NNI**。

## Azure 部署的 Kubernetes 的准备工作

1. NNI 支持基于 Azure Kubernetes Service 的 Kubeflow，参考[指南](https://azure.microsoft.com/en-us/services/kubernetes-service/)来设置 Azure Kubernetes Service。
2. 安装 [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) 和 **kubectl**。 使用 `az login` 命令来设置 Azure 账户吗，并将 kubectl 客户端连接到 AKS，参考此[指南](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster)。
3. 参考此[指南](https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal)来创建 Azure 文件存储账户。 NNI 需要 Azure Storage Service 来存取代码和输出文件。
4. To access Azure storage service, nni need the access key of the storage account, and nni use [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/) Service to protect your private key. Set up Azure Key Vault Service, add a secret to Key Vault to store the access key of Azure storage account. Follow this [guideline](https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli) to store the access key.

## Set up FrameworkController

Follow the [guideline](https://github.com/Microsoft/frameworkcontroller/tree/master/example/run) to set up frameworkcontroller in the kubernetes cluster, nni support frameworkcontroller by the statefulset mode.

## Design

Please refer the design of [kubeflow training service](./KubeflowMode.md), frameworkcontroller training service pipeline is similar.

## Example

The frameworkcontroller config file format is:

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
    

If you use Azure Kubernetes Service, you should set `frameworkcontrollerConfig` in your config yaml file as follows:

    frameworkcontrollerConfig:
      storage: azureStorage
      keyVault:
        vaultName: {your_vault_name}
        name: {your_secert_name}
      azureStorage:
        accountName: {your_storage_account_name}
        azureShare: {your_azure_share_name}
    

Note: You should explicitly set `trainingServicePlatform: frameworkcontroller` in nni config yaml file if you want to start experiment in frameworkcontrollerConfig mode.

The trial's config format for nni frameworkcontroller mode is a simple version of frameworkcontroller's offical config, you could refer the [tensorflow example of frameworkcontroller](https://github.com/Microsoft/frameworkcontroller/blob/master/example/framework/scenario/tensorflow/cpu/tensorflowdistributedtrainingwithcpu.yaml) for deep understanding.  
Trial configuration in frameworkcontroller mode have the following configuration keys:

* taskRoles: you could set multiple task roles in config file, and each task role is a basic unit to process in kubernetes cluster. 
   * name: the name of task role specified, like "worker", "ps", "master".
   * taskNum: the replica number of the task role.
   * command: the users' command to be used in the container.
   * gpuNum: the number of gpu device used in container.
   * cpuNum: the number of cpu device used in container.
   * memoryMB: the memory limitaion to be specified in container.
   * image: the docker image used to create pod and run the program.
   * frameworkAttemptCompletionPolicy: the policy to run framework, please refer the [user-manual](https://github.com/Microsoft/frameworkcontroller/blob/master/doc/user-manual.md#frameworkattemptcompletionpolicy) to get the specific information. Users could use the policy to control the pod, for example, if ps does not stop, only worker stops, this completionpolicy could helps stop ps.

## How to run example

After you prepare a config file, you could run your experiment by nnictl. The way to start an experiment on frameworkcontroller is similar to kubeflow, please refer the [document](./KubeflowMode.md) for more information.