# **在 Kubeflow 上运行实验。**

NNI 支持在 [Kubeflow](https://github.com/kubeflow/kubeflow)上运行，称为 kubeflow 模式。 在开始使用 NNI 的 kubeflow 模式前，需要有一个 kubernetes 集群，可以是私有部署的，或者是 [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/)，并需要一台配置好 [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) 的 Ubuntu 计算机连接到此 kubernetes 集群。 如果不熟悉 Kubernetes，可先浏览[这里](https://kubernetes.io/docs/tutorials/kubernetes-basics/)。 在 kubeflow 模式下，每个尝试程序会在 Kubernetes 集群中作为一个 kubeflow 作业来运行。

## 私有部署的 Kubernetes 的准备工作

1. 采用 Kubernetes 1.8 或更高版本。 根据[指南](https://kubernetes.io/docs/setup/)来安装 Kubernetes。
2. 在 Kubernetes 集群中下载、安装、部署 **Kubelow**。 根据[指南](https://www.kubeflow.org/docs/started/getting-started/)安装 Kubeflow。
3. 配置 **kubeconfig** 文件，NNI 将使用此配置与 Kubernetes API 服务交互。 默认情况下，NNI 管理器会使用 $(HOME)/.kube/config 作为 kubeconfig 文件的路径。 也可以通过环境变量 **KUBECONFIG** 来指定其它 kubeconfig 文件。 根据[指南](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig)了解更多 kubeconfig 的信息。 
4. 如果 NNI 尝试作业需要 GPU 资源，需按照[指南](https://github.com/NVIDIA/k8s-device-plugin)来配置 **Kubernetes 下的 Nvidia 插件**。
5. 准备 **NFS 服务器** 并导出通用的装载 (mount)，推荐将 NFS 服务器路径映射到 `root_squash 选项`，否则可能会在 NNI 复制文件到 NFS 时出现权限问题。 参考[页面](https://linux.die.net/man/5/exports)，来了解关于 root_squash 选项，或 **Azure File Storage**。 
6. 在安装 NNI 并运行 nnictl 的计算机上安装 **NFS 客户端**。 运行此命令安装 NFSv4 客户端：
    
        apt-get install nfs-common 
        

7. 参考[指南](GetStarted.md)安装 **NNI**。

## Azure 部署的 Kubernetes 的准备工作

1. NNI 支持基于 Azure Kubernetes Service 的 Kubeflow，参考[指南](https://azure.microsoft.com/en-us/services/kubernetes-service/)来设置 Azure Kubernetes Service。
2. 安装 [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) 和 **kubectl**。 使用 `az login` 命令来设置 Azure 账户吗，并将 kubectl 客户端连接到 AKS，参考此[指南](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster)。
3. 在 Azure Kubernetes Service 上部署 Kubeflow，参考此[指南](https://www.kubeflow.org/docs/started/getting-started/)。
4. 参考此[指南](https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal)来创建 Azure 文件存储账户。 NNI 需要 Azure Storage Service 来存取代码和输出文件。
5. NNI 需要访问密钥来连接 Azure 存储服务，NNI 使用 [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/) 服务来保护私钥。 设置 Azure Key Vault 服务，并添加密钥到 Key Vault 中来存取 Azure 存储账户。 参考[指南](https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli)来存储访问密钥。

## 设计

![](./img/kubeflow_training_design.png) Kubeflow 训练服务会实例化一个 kubernetes 客户端来与 Kubernetes 集群的 API 服务器交互。

对于每个尝试，会上传本机 codeDir 路径（在 nni_config.yaml 中配置）中的所有文件，包括 parameter.cfg 这样的生成的文件到存储卷中。 当前支持两种存储卷：[nfs](https://en.wikipedia.org/wiki/Network_File_System) 和 [Azure 文件存储](https://azure.microsoft.com/en-us/services/storage/files/)，需要在 NNI 的 yaml 文件中进行配置。 当文件准备好后，Kubeflow 训练服务会调用 Kubernetes 的 API 来创建 Kubeflow 作业 ([tf-operator](https://github.com/kubeflow/tf-operator) 作业或 [pytorch-operator](https://github.com/kubeflow/pytorch-operator) 作业) ，并将存储卷挂载到作业的 pod 中。 Kubeflow 作业的输出文件，例如 stdout, stderr, trial.log 以及模型文件，也会被复制回存储卷。 NNI 会在网页中显示每个尝试的存储卷的 URL，以便浏览日志和输出文件。

## 支持的操作符（operator）

NNI 仅支持 kubeflow 的 tf-operator 和 pytorch-operator，其它操作符未经测试。 可以在配置文件中设置操作符类型。 这是 tf-operator 的设置：

    kubeflowConfig:
      operator: tf-operator
    

这是 pytorch-operator 的设置：

    kubeflowConfig:
      operator: pytorch-operator
    

如果要使用 tf-operator，需要在尝试配置中设置 `ps` 和 `worker`。如果要使用 pytorch-operator，需要在尝试配置中设置 `master` 和 `worker`。

## Supported sotrage type

NNI support NFS and Azure Storage to store the code and output files, users could set storage type in config file and set the corresponding config.  
The setting for NFS storage are as follows:

    kubeflowConfig:
      storage: nfs
      nfs:
        # Your NFS server IP, like 10.10.10.10
        server: {your_nfs_server_ip}
        # Your NFS server export path, like /var/nfs/nni
        path: {your_nfs_server_export_path}
    

If you use Azure storage, you should set `kubeflowConfig` in your config yaml file as follows:

    kubeflowConfig:
      storage: azureStorage
      keyVault:
        vaultName: {your_vault_name}
        name: {your_secert_name}
      azureStorage:
        accountName: {your_storage_account_name}
        azureShare: {your_azure_share_name}
    

## Run an experiment

Use `examples/trials/mnist` as an example. This is a tensorflow job, and use tf-operator of kubeflow. The nni config yaml file's content is like:

    authorName: default
    experimentName: example_mnist
    trialConcurrency: 2
    maxExecDuration: 1h
    maxTrialNum: 20
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: kubeflow
    searchSpacePath: search_space.json
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
        # Your NFS server IP, like 10.10.10.10
        server: {your_nfs_server_ip}
        # Your NFS server export path, like /var/nfs/nni
        path: {your_nfs_server_export_path}
    

Note: You should explicitly set `trainingServicePlatform: kubeflow` in nni config yaml file if you want to start experiment in kubeflow mode.

If you want to run Pytorch jobs, you could set your config files as follow:

    authorName: default
    experimentName: example_mnist_distributed_pytorch
    trialConcurrency: 1
    maxExecDuration: 1h
    maxTrialNum: 10
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: kubeflow
    searchSpacePath: search_space.json
    #choice: true, false
    useAnnotation: false
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
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
        # Your NFS server IP, like 10.10.10.10
        server: {your_nfs_server_ip}
        # Your NFS server export path, like /var/nfs/nni
        path: {your_nfs_server_export_path}
    

Trial configuration in kubeflow mode have the following configuration keys:

* codeDir 
    * 代码目录，存放训练代码和配置文件
* worker (必填)。 此部分用于配置 TensorFlow 的 worker 角色 
    * replicas 
        * 必填。 需要运行的 TensorFlow woker 角色的数量，必须为正数。
    * command 
        * 必填。 用来运行尝试作业的命令，例如： ```python mnist.py```
    * memoryMB 
        * 必填。 尝试程序的内存需求，必须为正数。
    * cpuNum
    * gpuNum
    * image 
        * 必填。 在 kubeflow 模式中，Kubernetes 会安排尝试程序在 [Pod](https://kubernetes.io/docs/concepts/workloads/pods/pod/) 中执行。 此键用来指定尝试程序的 pod 使用的 Docker 映像。 
        * We already build a docker image [msranni/nni](https://hub.docker.com/r/msranni/nni/) on [Docker Hub](https://hub.docker.com/). 它包含了用来启动 NNI 实验所依赖的所有 Python 包，Node 模块和 JavaScript。 用来生成此映像的文件在[这里](../deployment/Dockerfile.build.base)。 可以直接使用此映像，或参考它来生成自己的映像。
    * apiVersion 
        * Required key. The API version of your kubeflow.
* ps (可选)。 此部分用于配置 TensorFlow 的 parameter 服务器角色。
* master(optional). This config section is used to configure pytorch parameter server role.

Once complete to fill nni experiment config file and save (for example, save as exp_kubeflow.yaml), then run the following command

    nnictl create --config exp_kubeflow.yaml
    

to start the experiment in kubeflow mode. NNI will create Kubeflow tfjob or pytorchjob for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. You can see the kubeflow tfjob created by NNI in your Kubernetes dashboard.

Notice: In kubeflow mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic.

Once a trial job is completed, you can goto NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information.

Any problems when using NNI in kubeflow mode, plesae create issues on [NNI github repo](https://github.com/Microsoft/nni), or send mail to nni@microsoft.com