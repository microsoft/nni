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

1. NNI support kubeflow based on Azure Kubernetes Service, follow the [guideline](https://azure.microsoft.com/en-us/services/kubernetes-service/) to set up Azure Kubernetes Service.
2. Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) and **kubectl**. Use `az login` to set azure account, and connect kubectl client to AKS, refer this [guideline](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster).
3. Deploy kubeflow on Azure Kubernetes Service, follow the [guideline](https://www.kubeflow.org/docs/started/getting-started/).
4. Follow the [guideline](https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal) to create azure file storage account. If you use Azure Kubernetes Service, nni need Azure Storage Service to store code files and the output files.
5. To access Azure storage service, nni need the access key of the storage account, and nni use [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/) Service to protect your private key. Set up Azure Key Vault Service, add a secret to Key Vault to store the access key of Azure storage account. Follow this [guideline](https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli) to store the access key.

## 设计

![](../../docs/img/kubeflow_training_design.png) Kubeflow training service instantiates a kubernetes rest client to interact with your K8s cluster's API server.

For each trial, we will upload all the files in your local codeDir path (configured in nni_config.yaml) together with NNI generated files like parameter.cfg into a storage volumn. Right now we support two kinds of storage volumns: [nfs](https://en.wikipedia.org/wiki/Network_File_System) and [azure file storage](https://azure.microsoft.com/en-us/services/storage/files/), you should configure the storage volumn in nni config yaml file. After files are prepared, Kubeflow training service will call K8S rest API to create kubeflow jobs ([tf-operator](https://github.com/kubeflow/tf-operator) job or [pytorch-operator](https://github.com/kubeflow/pytorch-operator) job) in K8S, and mount your storage volumn into the job's pod. Output files of kubeflow job, like stdout, stderr, trial.log or model files, will also be copied back to the storage volumn. NNI will show the storage volumn's URL for each trial in WebUI, to allow user browse the log files and job's output files.

## Run an experiment

Use `examples/trials/mnist` as an example. The nni config yaml file's content is like:

    authorName: your_name
    experimentName: example_mnist
    # how many trials could be concurrently running
    trialConcurrency: 4
    # maximum experiment running duration
    maxExecDuration: 3h
    # empty means never stop
    maxTrialNum: 100
    # choice: local, remote, pai, kubeflow
    trainingServicePlatform: kubeflow
    # choice: true, false  
    useAnnotation: false
    tuner:
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
        optimize_mode: maximize
    trial:
      codeDir: ~/nni/examples/trials/mnist
      ps:
        replicas: 1 
        command: python mnist-keras.py    
        gpuNum: 0
        cpuNum: 1
        memoryMB: 8196
        image: {your_docker_image_for_tensorflow_ps}
      worker:
        replicas: 1 
        command: python mnist-keras.py    
        gpuNum: 2
        cpuNum: 1
        memoryMB: 8196
        image: {your_docker_image_for_tensorflow_worker}
    kubeflowConfig:
      operator: tf-operator
      storage: nfs
      nfs:
        server: {your_nfs_server}
        path: {your_nfs_server_exported_path}
    

If you use Azure Kubernetes Service, you should set `kubeflowConfig` in your config yaml file as follows:

    kubeflowConfig:
      operator: tf-operator
      storage: azureStorage
      keyVault:
        vaultName: {your_vault_name}
        name: {your_secert_name}
      azureStorage:
        accountName: {your_storage_account_name}
        azureShare: {your_azure_share_name}
    

Note: You should explicitly set `trainingServicePlatform: kubeflow` in nni config yaml file if you want to start experiment in kubeflow mode.

Trial configuration in kubeflow mode have the following configuration keys:

* codeDir 
    * code directory, where you put training code and config files
* worker (required). This config section is used to configure tensorflow worker role 
    * replicas 
        * Required key. Should be positive number depends on how many replication your want to run for tensorflow worker role.
    * command 
        * Required key. Command to launch your trial job, like ```python mnist.py```
    * memoryMB 
        * Required key. Should be positive number based on your trial program's memory requirement
    * cpuNum
    * gpuNum
    * image 
        * Required key. In kubeflow mode, your trial program will be scheduled by Kubernetes to run in [Pod](https://kubernetes.io/docs/concepts/workloads/pods/pod/). This key is used to specify the Docker image used to create the pod where your trail program will run. 
        * We already build a docker image [nnimsra/nni](https://hub.docker.com/r/msranni/nni/) on [Docker Hub](https://hub.docker.com/). It contains NNI python packages, Node modules and javascript artifact files required to start experiment, and all of NNI dependencies. The docker file used to build this image can be found at [here](../deployment/Dockerfile.build.base). You can either use this image directly in your config file, or build your own image based on it.
* ps (optional). This config section is used to configure tensorflow parameter server role.

Once complete to fill nni experiment config file and save (for example, save as exp_kubeflow.yaml), then run the following command

    nnictl create --config exp_kubeflow.yaml
    

to start the experiment in kubeflow mode. NNI will create Kubeflow tfjob for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. You can see the kubeflow tfjob created by NNI in your Kubernetes dashboard.

Notice: In kubeflow mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic.

Once a trial job is completed, you can goto NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information.

Any problems when using NNI in kubeflow mode, plesae create issues on [NNI github repo](https://github.com/Microsoft/nni), or send mail to nni@microsoft.com