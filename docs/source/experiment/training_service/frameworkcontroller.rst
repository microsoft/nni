FrameworkController Training Service
====================================

NNI supports running experiment using `FrameworkController <https://github.com/Microsoft/frameworkcontroller>`__,
called frameworkcontroller mode.
FrameworkController is built to orchestrate all kinds of applications on Kubernetes,
you don't need to install Kubeflow for specific deep learning framework like tf-operator or pytorch-operator.
Now you can use FrameworkController as the training service to run NNI experiment.

Prerequisite for on-premises Kubernetes Service
-----------------------------------------------

1. A **Kubernetes** cluster using Kubernetes 1.8 or later.
   Follow this `guideline <https://kubernetes.io/docs/setup/>`__ to set up Kubernetes.
2. Prepare a **kubeconfig** file, which will be used by NNI to interact with your Kubernetes API server.
   By default, NNI manager will use ``~/.kube/config`` as kubeconfig file's path.
   You can also specify other kubeconfig files by setting the**KUBECONFIG** environment variable.
   Refer this `guideline <https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig>`__
   to learn more about kubeconfig.
3. If your NNI trial job needs GPU resource, you should follow this `guideline <https://github.com/NVIDIA/k8s-device-plugin>`__
   to configure **Nvidia device plugin for Kubernetes**.
4. Prepare a **NFS server** and export a general purpose mount
   (we recommend to map your NFS server path in ``root_squash option``,
   otherwise permission issue may raise when NNI copies files to NFS.
   Refer this `page <https://linux.die.net/man/5/exports>`__ to learn what root_squash option is),
   or **Azure File Storage**.
5. Install **NFS client** on the machine where you install NNI and run nnictl to create experiment.
   Run this command to install NFSv4 client:

.. code-block:: bash

    apt install nfs-common

6. Install **NNI**:

.. code-block:: bash

    python -m pip install nni

Prerequisite for Azure Kubernetes Service
-----------------------------------------

1. NNI support FrameworkController based on Azure Kubernetes Service,
   follow the `guideline <https://azure.microsoft.com/en-us/services/kubernetes-service/>`__ to set up Azure Kubernetes Service.
2. Install `Azure CLI <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest>`__ and **kubectl**.
   Use ``az login`` to set azure account, and connect kubectl client to AKS,
   refer this `guideline <https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster>`__.
3. Follow the `guideline <https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal>`__
   to create azure file storage account.
   If you use Azure Kubernetes Service, NNI need Azure Storage Service to store code files and the output files.
4. To access Azure storage service, NNI need the access key of the storage account,
   and NNI uses `Azure Key Vault <https://azure.microsoft.com/en-us/services/key-vault/>`__ Service to protect your private key.
   Set up Azure Key Vault Service, add a secret to Key Vault to store the access key of Azure storage account.
   Follow this `guideline <https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli>`__ to store the access key.

Setup FrameworkController
-------------------------

Follow the `guideline <https://github.com/Microsoft/frameworkcontroller/tree/master/example/run>`__
to set up FrameworkController in the Kubernetes cluster, NNI supports FrameworkController by the stateful set mode.
If your cluster enforces authorization, you need to create a service account with granted permission for FrameworkController,
and then pass the name of the FrameworkController service account to the NNI Experiment Config.
`refer <https://github.com/Microsoft/frameworkcontroller/tree/master/example/run#run-by-kubernetes-statefulset>`__.  
If the k8s cluster enforces Authorization, you also need to create a ServiceAccount with granted permission for FrameworkController,
`refer <https://github.com/microsoft/frameworkcontroller/tree/master/example/run#prerequisite>`__.  

Design
------

Please refer the design of `Kubeflow training service <KubeflowMode.rst>`__,
FrameworkController training service pipeline is similar.

Example
-------

The FrameworkController config format is:

.. code-block:: python

    from nni.experiment import (
        Experiment,
        FrameworkAttemptCompletionPolicy,
        FrameworkControllerRoleConfig,
        K8sNfsConfig,
    )

    experiment = Experiment('frameworkcontroller')
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 10
    experiment.config.trial_concurrency = 2

    experiment.config.training_service.storage = K8sNfsConfig()
    experiment.config.training_service.storage.server = '10.20.30.40'
    experiment.config.training_service.storage.path = '/mnt/nfs/nni'
    experiment.config.training_service.task_roles = [FrameworkControllerRoleConfig()]
    experiment.config.training_service.task_roles[0].name = 'worker'
    experiment.config.training_service.task_roles[0].task_number = 1
    experiment.config.training_service.task_roles[0].command = 'python3 model.py'
    experiment.config.training_service.task_roles[0].gpuNumber = 1
    experiment.config.training_service.task_roles[0].cpuNumber = 1
    experiment.config.training_service.task_roles[0].memorySize = '4g'
    experiment.config.training_service.task_roles[0].framework_attempt_completion_policy = \
        FrameworkAttemptCompletionPolicy(min_failed_task_count = 1, min_succeed_task_count = 1)
        
If you use Azure Kubernetes Service, you should set storage config as follows:

.. code-block:: python

    experiment.config.training_service.storage = K8sAzureStorageConfig()
    experiment.config.training_service.storage.azure_account = 'your_storage_account_name'
    experiment.config.training_service.storage.azure_share = 'your_azure_share_name'
    experiment.config.training_service.storage.key_vault_name = 'your_vault_name'
    experiment.config.training_service.storage.key_vault_key = 'your_secret_name'

If you set `ServiceAccount <https://github.com/microsoft/frameworkcontroller/tree/master/example/run#prerequisite>`__ in your k8s,
please set ``serviceAccountName`` in your config:

.. code-block:: python

    experiment.config.training_service.service_account_name = 'frameworkcontroller'

The trial's config format for NNI frameworkcontroller mode is a simple version of FrameworkController's official config,
you could refer the `Tensorflow example of FrameworkController
<https://github.com/microsoft/frameworkcontroller/blob/master/example/framework/scenario/tensorflow/ps/cpu/tensorflowdistributedtrainingwithcpu.yaml>`__
for deep understanding.

Once it's ready, run:

.. code-block:: python

    experiment.run(8080)

Notice: In frameworkcontroller mode,
NNIManager will start a rest server and listen on a port which is your NNI web portal's port plus 1.
For example, if your web portal port is ``8080``, the rest server will listen on ``8081``,
to receive metrics from trial job running in Kubernetes.
So you should ``enable 8081`` TCP port in your firewall rule to allow incoming traffic.
