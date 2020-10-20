# 在 Kubeflow 上运行 Experiment

===

NNI 支持在 [Kubeflow](https://github.com/kubeflow/kubeflow)上运行，称为 kubeflow 模式。 在开始使用 NNI 的 Kubeflow 模式前，需要有一个 Kubernetes 集群，可以是私有部署的，或者是 [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/zh-cn/services/kubernetes-service/)，并需要一台配置好 [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) 的 Ubuntu 计算机连接到此 Kubernetes 集群。 如果不熟悉 Kubernetes，可先浏览[这里](https://kubernetes.io/docs/tutorials/kubernetes-basics/)。 在 kubeflow 模式下，每个 Trial 程序会在 Kubernetes 集群中作为一个 Kubeflow 作业来运行。

## 私有部署的 Kubernetes 的准备工作

1. 采用 Kubernetes 1.8 或更高版本。 根据[指南](https://kubernetes.io/docs/setup/)来安装 Kubernetes。
2. 在 Kubernetes 集群中下载、安装、部署 **Kubeflow**。 根据[指南](https://www.kubeflow.org/docs/started/getting-started/)安装 Kubeflow。
3. 配置 **kubeconfig** 文件，NNI 将使用此配置与 Kubernetes API 服务交互。 默认情况下，NNI 管理器会使用 $(HOME)/.kube/config 作为 kubeconfig 文件的路径。 也可以通过环境变量 **KUBECONFIG** 来指定其它 kubeconfig 文件。 根据[指南](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig)了解更多 kubeconfig 的信息。
4. 如果 NNI Trial 作业需要 GPU 资源，需按照[指南](https://github.com/NVIDIA/k8s-device-plugin)来配置 **Kubernetes 下的 Nvidia 插件**。
5. 准备 **NFS 服务器** 并导出通用的装载 (mount)，推荐将 NFS 服务器路径映射到 `root_squash 选项`，否则可能会在 NNI 复制文件到 NFS 时出现权限问题。 参考[页面](https://linux.die.net/man/5/exports)，来了解关于 root_squash 选项，或 **Azure File Storage**。
6. 在安装 NNI 并运行 nnictl 的计算机上安装 **NFS 客户端**。 运行此命令安装 NFSv4 客户端： ```apt-get install nfs-common```

7. 参考[指南](../Tutorial/QuickStart.md)安装 **NNI**。

## Azure 部署的 Kubernetes 的准备工作

1. NNI 支持基于 Azure Kubernetes Service 的 Kubeflow，参考[指南](https://azure.microsoft.com/zh-cn/services/kubernetes-service/)来设置 Azure Kubernetes Service。
2. 安装 [Azure CLI](https://docs.microsoft.com/zh-cn/cli/azure/install-azure-cli?view=azure-cli-latest) 和 **kubectl**。 使用 `az login` 命令来设置 Azure 账户吗，并将 kubectl 客户端连接到 AKS，参考此[指南](https://docs.microsoft.com/zh-cn/azure/aks/kubernetes-walkthrough#connect-to-the-cluster)。
3. 在 Azure Kubernetes Service 上部署 Kubeflow，参考此[指南](https://www.kubeflow.org/docs/started/getting-started/)。
4. 参考此[指南](https://docs.microsoft.com/zh-cn/azure/storage/common/storage-quickstart-create-account?tabs=portal)来创建 Azure 文件存储账户。 NNI 需要 Azure Storage Service 来存取代码和输出文件。
5. NNI 需要访问密钥来连接 Azure 存储服务，NNI 使用 [Azure Key Vault](https://azure.microsoft.com/zh-cn/services/key-vault/) 服务来保护私钥。 设置 Azure Key Vault 服务，并添加密钥到 Key Vault 中来存取 Azure 存储账户。 参考[指南](https://docs.microsoft.com/zh-cn/azure/key-vault/quick-create-cli)来存储访问密钥。

## 设计

![](../../img/kubeflow_training_design.png) Kubeflow 训练平台会实例化一个 Kubernetes 客户端来与 Kubernetes 集群的 API 服务器交互。

对于每个 Trial，会上传本机 codeDir 路径（在 nni_config.yml 中配置）中的所有文件，包括 parameter.cfg 这样的生成的文件到存储卷中。 当前支持两种存储卷：[nfs](https://en.wikipedia.org/wiki/Network_File_System) 和 [Azure 文件存储](https://azure.microsoft.com/zh-cn/services/storage/files/)，需要在 NNI 的 YAML 文件中进行配置。 当文件准备好后，Kubeflow 训练平台会调用 Kubernetes 的 API 来创建 Kubeflow 作业 ([tf-operator](https://github.com/kubeflow/tf-operator) 作业或 [pytorch-operator](https://github.com/kubeflow/pytorch-operator) 作业) ，并将存储卷挂载到作业的 pod 中。 Kubeflow 作业的输出文件，例如 stdout, stderr, trial.log 以及模型文件，也会被复制回存储卷。 NNI 会在网页中显示每个 Trial 的存储卷的 URL，以便浏览日志和输出文件。

## 支持的操作符（operator）

NNI 仅支持 Kubeflow 的 tf-operator 和 pytorch-operator，其它操作符未经测试。 可以在配置文件中设置操作符类型。 这是 tf-operator 的设置：

```yaml
kubeflowConfig:
  operator: tf-operator
```

这是 pytorch-operator 的设置：

```yaml
kubeflowConfig:
  operator: pytorch-operator
```

如果要使用 tf-operator，需要在 Trial 配置中设置 `ps` 和 `worker`。如果要使用 pytorch-operator，需要在 Trial 配置中设置 `master` 和 `worker`。

## 支持的存储类型

NNI 支持使用 NFS 和 Azure 存储来存储代码和输出文件，可在配置文件进行相应的配置。

NFS 存储配置如下：

```yaml
kubeflowConfig:
  storage: nfs
  nfs:
    # NFS 服务器 IP， 如 10.10.10.10
    server: {your_nfs_server_ip}
    # NFS 服务器的导出路径，如 /var/nfs/nni
    path: {your_nfs_server_export_path}
```

如果使用了 Azure 存储，需要在 YAML 文件中如下设置 `kubeflowConfig`：

```yaml
kubeflowConfig:
  storage: azureStorage
  keyVault:
    vaultName: {your_vault_name}
    name: {your_secert_name}
  azureStorage:
    accountName: {your_storage_account_name}
    azureShare: {your_azure_share_name}
```

## 运行 Experiment

以 `examples/trials/mnist-tfv1` 为例。 这是一个 TensorFlow 作业，使用了 Kubeflow 的 tf-operator。 NNI 的 YAML 配置文件如下：

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 2
maxExecDuration: 1h
maxTrialNum: 20
#可选项: local, remote, pai, kubeflow
trainingServicePlatform: kubeflow
searchSpacePath: search_space.json
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
  codeDir: .
  worker:
    replicas: 2
    command: python3 dist_mnist.py
    gpuNum: 1
    cpuNum: 1
    memoryMB: 8196
    image: msranni/nni:latest
  ps:
    replicas: 1
    command: python3 dist_mnist.py
    gpuNum: 0
    cpuNum: 1
    memoryMB: 8196
    image: msranni/nni:latest
kubeflowConfig:
  operator: tf-operator
  apiVersion: v1alpha2
  storage: nfs
  nfs:
    # NFS 服务器 IP，如 10.10.10.10
    server: {your_nfs_server_ip}
    # NFS 服务器的导出路径，如 /var/nfs/nni
    path: {your_nfs_server_export_path}
```

注意：如果用 Kubeflow 模式运行，需要在 YAML 文件中显式设置 `trainingServicePlatform: kubeflow`。

如果要运行 Pytorch 作业，需要如下配置：

```yaml
authorName: default
experimentName: example_mnist_distributed_pytorch
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#可选项: local, remote, pai, kubeflow
trainingServicePlatform: kubeflow
searchSpacePath: search_space.json
#可选项: true, false
useAnnotation: false
tuner:
  #可选项: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: minimize
trial:
  codeDir: .
  master:
    replicas: 1
    command: python3 dist_mnist.py
    gpuNum: 1
    cpuNum: 1
    memoryMB: 2048
    image: msranni/nni:latest
  worker:
    replicas: 1
    command: python3 dist_mnist.py
    gpuNum: 0
    cpuNum: 1
    memoryMB: 2048
    image: msranni/nni:latest
kubeflowConfig:
  operator: pytorch-operator
  apiVersion: v1alpha2
  nfs:
    # NFS 服务器 IP，如 10.10.10.10
    server: {your_nfs_server_ip}
    # NFS 服务器导出路径，如  /var/nfs/nni
    path: {your_nfs_server_export_path}
```

Kubeflow 模式的配置有下列主键：

* codeDir 
  * 代码目录，存放训练代码和配置文件
* worker (必填)。 此部分用于配置 TensorFlow 的 worker 角色 
  * replicas 
    * 必填。 需要运行的 TensorFlow woker 角色的数量，必须为正数。
  * command 
    * 必填。 用来运行 Trial 作业的命令，例如： ```python mnist.py```
  * memoryMB 
    * 必填。 Trial 程序的内存需求，必须为正数。
  * cpuNum
  * gpuNum
  * image 
    * 必填。 在 kubeflow 模式中，Kubernetes 会安排 Trial 程序在 [Pod](https://kubernetes.io/docs/concepts/workloads/pods/pod/) 中执行。 此键用来指定 Trial 程序的 pod 使用的 Docker 映像。
    * [Docker Hub](https://hub.docker.com/) 上有预制的 NNI Docker 映像 [msranni/nni](https://hub.docker.com/r/msranni/nni/)。 它包含了用来启动 NNI Experiment 所依赖的所有 Python 包，Node 模块和 JavaScript。 生成此 Docker 映像的文件在[这里](https://github.com/Microsoft/nni/tree/master/deployment/docker/Dockerfile)。 可以直接使用此映像，或参考它来生成自己的映像。
  * privateRegistryAuthPath 
    * 可选字段，指定 `config.json` 文件路径。此文件，包含了 Docker 注册的认证令牌，用来从私有 Docker 中拉取映像。 [参考文档](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/)。
  * apiVersion 
    * 必填。 Kubeflow 的 API 版本。
* ps (可选)。 此部分用于配置 TensorFlow 的 parameter 服务器角色。
* master (可选)。 此部分用于配置 PyTorch 的 parameter 服务器角色。

完成并保存 NNI Experiment 配置文件后（例如可保存为：exp_kubeflow.yml），运行以下命令：

```bash
nnictl create --config exp_kubeflow.yml
```

来在 Kubeflow 模式下启动 Experiment。 NNI 会为每个 Trial 创建 Kubeflow tfjob 或 pytorchjob，作业名称的格式为 `nni_exp_{experiment_id}_trial_{trial_id}`。 可以在 Kubernetes 面板中看到创建的 Kubeflow tfjob。

注意：Kubeflow 模式下，NNIManager 会启动 RESTful 服务，监听端口为 NNI 网页服务器的端口加1。 例如，如果网页端口为`8080`，那么 RESTful 服务器会监听在 `8081`端口，来接收运行在 Kubernetes 中的 Trial 作业的指标。 因此，需要在防火墙中启用端口 `8081` 的 TCP 协议，以允许传入流量。

当一个 Trial 作业完成后，可以在 NNI 网页的概述页面（如：http://localhost:8080/oview）中查看 Trial 的信息。

## 版本校验

从 0.6 开始，NNI 支持版本校验，详情参考[这里](PaiMode.md)。

如果在使用 Kubeflow 模式时遇到任何问题，请到 [NNI Github](https://github.com/Microsoft/nni) 中创建问题。