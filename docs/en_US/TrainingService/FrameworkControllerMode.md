# Run an Experiment on FrameworkController

===
NNI supports running experiment using [FrameworkController](https://github.com/Microsoft/frameworkcontroller), called frameworkcontroller mode. FrameworkController is built to orchestrate all kinds of applications on Kubernetes, you don't need to install Kubeflow for specific deep learning framework like tf-operator or pytorch-operator. Now you can use FrameworkController as the training service to run NNI experiment.

## Prerequisite for on-premises Kubernetes Service

1. A **Kubernetes** cluster using Kubernetes 1.8 or later. Follow this [guideline](https://kubernetes.io/docs/setup/) to set up Kubernetes
2. Prepare a **kubeconfig** file, which will be used by NNI to interact with your Kubernetes API server. By default, NNI manager will use $(HOME)/.kube/config as kubeconfig file's path. You can also specify other kubeconfig files by setting the **KUBECONFIG** environment variable. Refer this [guideline]( https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig) to learn more about kubeconfig.
3. If your NNI trial job needs GPU resource, you should follow this [guideline](https://github.com/NVIDIA/k8s-device-plugin) to configure **Nvidia device plugin for Kubernetes**.
4. Prepare a **NFS server** and export a general purpose mount (we recommend to map your NFS server path in `root_squash option`, otherwise permission issue may raise when NNI copies files to NFS. Refer this [page](https://linux.die.net/man/5/exports) to learn what root_squash option is), or **Azure File Storage**.
5. Install **NFS client** on the machine where you install NNI and run nnictl to create experiment. Run this command to install NFSv4 client:

    ```bash
    apt-get install nfs-common
    ```

6. Install **NNI**, follow the install guide [here](../Tutorial/QuickStart.md).

## Prerequisite for Azure Kubernetes Service

1. NNI support Kubeflow based on Azure Kubernetes Service, follow the [guideline](https://azure.microsoft.com/en-us/services/kubernetes-service/) to set up Azure Kubernetes Service.
2. Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) and __kubectl__.  Use `az login` to set azure account, and connect kubectl client to AKS, refer this [guideline](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster).
3. Follow the [guideline](https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal) to create azure file storage account. If you use Azure Kubernetes Service, NNI need Azure Storage Service to store code files and the output files.
4. To access Azure storage service, NNI need the access key of the storage account, and NNI uses [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/) Service to protect your private key. Set up Azure Key Vault Service, add a secret to Key Vault to store the access key of Azure storage account. Follow this [guideline](https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli) to store the access key.

## Setup FrameworkController

Follow the [guideline](https://github.com/Microsoft/frameworkcontroller/tree/master/example/run) to set up FrameworkController in the Kubernetes cluster, NNI supports FrameworkController by the stateful set mode. If your cluster enforces authorization, you need to create a service account with granted permission for FrameworkController, and then pass the name of the FrameworkController service account to the NNI Experiment Config. [refer](https://github.com/Microsoft/frameworkcontroller/tree/master/example/run#run-by-kubernetes-statefulset)

## Design

Please refer the design of [Kubeflow training service](KubeflowMode.md), FrameworkController training service pipeline is similar.

## Example

The FrameworkController config file format is:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 10h
maxTrialNum: 100
#choice: local, remote, pai, kubeflow, frameworkcontroller
trainingServicePlatform: frameworkcontroller
searchSpacePath: ~/nni/examples/trials/mnist-tfv1/search_space.json
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

If you use Azure Kubernetes Service, you should  set `frameworkcontrollerConfig` in your config YAML file as follows:

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

Note: You should explicitly set `trainingServicePlatform: frameworkcontroller` in NNI config YAML file if you want to start experiment in frameworkcontrollerConfig mode.

The trial's config format for NNI frameworkcontroller mode is a simple version of FrameworkController's official config, you could refer the [Tensorflow example of FrameworkController](https://github.com/Microsoft/frameworkcontroller/blob/master/example/framework/scenario/tensorflow/cpu/tensorflowdistributedtrainingwithcpu.yaml) for deep understanding.

Trial configuration in frameworkcontroller mode have the following configuration keys:

* taskRoles: you could set multiple task roles in config file, and each task role is a basic unit to process in Kubernetes cluster.
  * name: the name of task role specified, like "worker", "ps", "master".
  * taskNum: the replica number of the task role.
  * command: the users' command to be used in the container.
  * gpuNum: the number of gpu device used in container.
  * cpuNum: the number of cpu device used in container.
  * memoryMB: the memory limitaion to be specified in container.
  * image: the docker image used to create pod and run the program.
  * frameworkAttemptCompletionPolicy: the policy to run framework, please refer the [user-manual](https://github.com/Microsoft/frameworkcontroller/blob/master/doc/user-manual.md#frameworkattemptcompletionpolicy) to get the specific information. Users could use the policy to control the pod, for example, if ps does not stop, only worker stops, The completion policy could helps stop ps.

## How to run example

After you prepare a config file, you could run your experiment by nnictl. The way to start an experiment on FrameworkController is similar to Kubeflow, please refer the [document](KubeflowMode.md) for more information.

## version check

NNI support version check feature in since version 0.6, [refer](PaiMode.md)
