# 在 Kubeflow 上运行 Experiment

===

NNI 支持在 [Kubeflow](https://github.com/kubeflow/kubeflow)上运行，称为 kubeflow 模式。 在开始使用 NNI 的 Kubeflow 模式前，需要有一个 Kubernetes 集群，可以是私有部署的，或者是 [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/)，并需要一台配置好 [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) 的 Ubuntu 计算机连接到此 Kubernetes 集群。 如果不熟悉 Kubernetes，可先浏览[这里](https://kubernetes.io/docs/tutorials/kubernetes-basics/)。 在 kubeflow 模式下，每个 Trial 程序会在 Kubernetes 集群中作为一个 Kubeflow 作业来运行。

## 私有部署的 Kubernetes 的准备工作

1. 采用 Kubernetes 1.8 或更高版本。 根据[指南](https://kubernetes.io/docs/setup/)来安装 Kubernetes。
2. 在 Kubernetes 集群中下载、安装、部署 **Kubeflow**。 根据[指南](https://www.kubeflow.org/docs/started/getting-started/)安装 Kubeflow。
3. 配置 **kubeconfig** 文件，NNI 将使用此配置与 Kubernetes API 服务交互。 默认情况下，NNI 管理器会使用 $(HOME)/.kube/config 作为 kubeconfig 文件的路径。 也可以通过环境变量 **KUBECONFIG** 来指定其它 kubeconfig 文件。 根据[指南](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig)了解更多 kubeconfig 的信息。
4. 如果 NNI Trial 作业需要 GPU 资源，需按照[指南](https://github.com/NVIDIA/k8s-device-plugin)来配置 **Kubernetes 下的 Nvidia 插件**。
5. 准备 **NFS 服务器** 并导出通用的装载 (mount)，推荐将 NFS 服务器路径映射到 `root_squash 选项`，否则可能会在 NNI 复制文件到 NFS 时出现权限问题。 参考[页面](https://linux.die.net/man/5/exports)，来了解关于 root_squash 选项，或 **Azure File Storage**。
6. 在安装 NNI 并运行 nnictl 的计算机上安装 **NFS 客户端**。 运行此命令安装 NFSv4 客户端： ```apt-get install nfs-common```

7. 参考[指南](QuickStart.md)安装 **NNI**。

## Azure 部署的 Kubernetes 的准备工作

1. NNI support Kubeflow based on Azure Kubernetes Service, follow the [guideline](https://azure.microsoft.com/en-us/services/kubernetes-service/) to set up Azure Kubernetes Service.
2. 安装 [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) 和 **kubectl**。 使用 `az login` 命令来设置 Azure 账户吗，并将 kubectl 客户端连接到 AKS，参考此[指南](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster)。
3. Deploy Kubeflow on Azure Kubernetes Service, follow the [guideline](https://www.kubeflow.org/docs/started/getting-started/).
4. 参考此[指南](https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal)来创建 Azure 文件存储账户。 NNI 需要 Azure Storage Service 来存取代码和输出文件。
5. NNI 需要访问密钥来连接 Azure 存储服务，NNI 使用 [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/) 服务来保护私钥。 设置 Azure Key Vault 服务，并添加密钥到 Key Vault 中来存取 Azure 存储账户。 参考[指南](https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli)来存储访问密钥。

## 设计

![](../img/kubeflow_training_design.png) Kubeflow training service instantiates a Kubernetes rest client to interact with your K8s cluster's API server.

For each trial, we will upload all the files in your local codeDir path (configured in nni_config.yml) together with NNI generated files like parameter.cfg into a storage volumn. Right now we support two kinds of storage volumes: [nfs](https://en.wikipedia.org/wiki/Network_File_System) and [azure file storage](https://azure.microsoft.com/en-us/services/storage/files/), you should configure the storage volumn in NNI config YAML file. After files are prepared, Kubeflow training service will call K8S rest API to create Kubeflow jobs ([tf-operator](https://github.com/kubeflow/tf-operator) job or [pytorch-operator](https://github.com/kubeflow/pytorch-operator) job) in K8S, and mount your storage volume into the job's pod. Output files of Kubeflow job, like stdout, stderr, trial.log or model files, will also be copied back to the storage volumn. NNI will show the storage volumn's URL for each trial in WebUI, to allow user browse the log files and job's output files.

## 支持的操作符（operator）

NNI only support tf-operator and pytorch-operator of Kubeflow, other operators is not tested. Users could set operator type in config file. The setting of tf-operator:

```yaml
kubeflowConfig:
  operator: tf-operator
```

The setting of pytorch-operator:

```yaml
kubeflowConfig:
  operator: pytorch-operator
```

If users want to use tf-operator, he could set `ps` and `worker` in trial config. If users want to use pytorch-operator, he could set `master` and `worker` in trial config.

## 支持的存储类型

NNI support NFS and Azure Storage to store the code and output files, users could set storage type in config file and set the corresponding config.

The setting for NFS storage are as follows:

```yaml
kubeflowConfig:
  storage: nfs
  nfs:
    # NFS 服务器 IP， 如 10.10.10.10
    server: {your_nfs_server_ip}
    # NFS 服务器的导出路径，如 /var/nfs/nni
    path: {your_nfs_server_export_path}
```

If you use Azure storage, you should set `kubeflowConfig` in your config YAML file as follows:

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

Use `examples/trials/mnist` as an example. This is a tensorflow job, and use tf-operator of Kubeflow. The NNI config YAML file's content is like:

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
  gpuNum: 0
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

Note: You should explicitly set `trainingServicePlatform: kubeflow` in NNI config YAML file if you want to start experiment in kubeflow mode.

If you want to run PyTorch jobs, you could set your config files as follow:

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

Trial configuration in kubeflow mode have the following configuration keys:

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
  * apiVersion 
    * 必填。 The API version of your Kubeflow.
* ps (可选)。 This config section is used to configure Tensorflow parameter server role.
* master (可选)。 This config section is used to configure PyTorch parameter server role.

Once complete to fill NNI experiment config file and save (for example, save as exp_kubeflow.yml), then run the following command

```bash
nnictl create --config exp_kubeflow.yml
```

to start the experiment in kubeflow mode. NNI will create Kubeflow tfjob or pytorchjob for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. You can see the Kubeflow tfjob created by NNI in your Kubernetes dashboard.

Notice: In kubeflow mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic.

Once a trial job is completed, you can go to NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information.

## 版本校验

NNI support version check feature in since version 0.6, [refer](PaiMode.md)

Any problems when using NNI in Kubeflow mode, please create issues on [NNI Github repo](https://github.com/Microsoft/nni).