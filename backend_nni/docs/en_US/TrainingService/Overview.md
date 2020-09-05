# Training Service

## What is Training Service?

NNI training service is designed to allow users to focus on AutoML itself, agnostic to the underlying computing infrastructure where the trials are actually run. When migrating from one cluster to another (e.g., local machine to Kubeflow), users only need to tweak several configurations, and the experiment can be easily scaled.

Users can use training service provided by NNI, to run trial jobs on [local machine](./LocalMode.md), [remote machines](./RemoteMachineMode.md), and on clusters like [PAI](./PaiMode.md), [Kubeflow](./KubeflowMode.md), [FrameworkController](./FrameworkControllerMode.md), [DLTS](./DLTSMode.md) and [AML](./AMLMode.md). These are called *built-in training services*.

If the computing resource customers try to use is not listed above, NNI provides interface that allows users to build their own training service easily. Please refer to "[how to implement training service](./HowToImplementTrainingService)" for details.

## How to use Training Service?

Training service needs to be chosen and configured properly in experiment configuration YAML file. Users could refer to the document of each training service for how to write the configuration. Also, [reference](../Tutorial/ExperimentConfig) provides more details on the specification of the experiment configuration file.

Next, users should prepare code directory, which is specified as `codeDir` in config file. Please note that in non-local mode, the code directory will be uploaded to remote or cluster before the experiment. Therefore, we limit the number of files to 2000 and total size to 300MB. If the code directory contains too many files, users can choose which files and subfolders should be excluded by adding a `.nniignore` file that works like a `.gitignore` file. For more details on how to write this file, see [this example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/.nniignore) and the [git documentation](https://git-scm.com/docs/gitignore#_pattern_format).

In case users intend to use large files in their experiment (like large-scaled datasets) and they are not using local mode, they can either: 1) download the data before each trial launches by putting it into trial command; or 2) use a shared storage that is accessible to worker nodes. Usually, training platforms are equipped with shared storage, and NNI allows users to easily use them. Refer to docs of each built-in training service for details.

## Built-in Training Services

|TrainingService|Brief Introduction|
|---|---|
|[__Local__](./LocalMode.md)|NNI supports running an experiment on local machine, called local mode. Local mode means that NNI will run the trial jobs and nniManager process in same machine, and support gpu schedule function for trial jobs.|
|[__Remote__](./RemoteMachineMode.md)|NNI supports running an experiment on multiple machines through SSH channel, called remote mode. NNI assumes that you have access to those machines, and already setup the environment for running deep learning training code. NNI will submit the trial jobs in remote machine, and schedule suitable machine with enough gpu resource if specified.|
|[__PAI__](./PaiMode.md)|NNI supports running an experiment on [OpenPAI](https://github.com/Microsoft/pai) (aka PAI), called PAI mode. Before starting to use NNI PAI mode, you should have an account to access an [OpenPAI](https://github.com/Microsoft/pai) cluster. See [here](https://github.com/Microsoft/pai#how-to-deploy) if you don't have any OpenPAI account and want to deploy an OpenPAI cluster. In PAI mode, your trial program will run in PAI's container created by Docker.|
|[__Kubeflow__](./KubeflowMode.md)|NNI supports running experiment on [Kubeflow](https://github.com/kubeflow/kubeflow), called kubeflow mode. Before starting to use NNI kubeflow mode, you should have a Kubernetes cluster, either on-premises or [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/), a Ubuntu machine on which [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) is setup to connect to your Kubernetes cluster. If you are not familiar with Kubernetes, [here](https://kubernetes.io/docs/tutorials/kubernetes-basics/) is a good start. In kubeflow mode, your trial program will run as Kubeflow job in Kubernetes cluster.|
|[__FrameworkController__](./FrameworkControllerMode.md)|NNI supports running experiment using [FrameworkController](https://github.com/Microsoft/frameworkcontroller), called frameworkcontroller mode. FrameworkController is built to orchestrate all kinds of applications on Kubernetes, you don't need to install Kubeflow for specific deep learning framework like tf-operator or pytorch-operator. Now you can use FrameworkController as the training service to run NNI experiment.|
|[__DLTS__](./DLTSMode.md)|NNI supports running experiment using [DLTS](https://github.com/microsoft/DLWorkspace.git), which is an open source toolkit, developed by Microsoft, that allows AI scientists to spin up an AI cluster in turn-key fashion.|
|[__AML__](./AMLMode.md)|NNI supports running an experiment on [AML](https://azure.microsoft.com/en-us/services/machine-learning/) , called aml mode.

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
