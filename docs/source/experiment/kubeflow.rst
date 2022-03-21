Kubeflow Training Service
=========================

Now NNI supports running experiment on `Kubeflow <https://github.com/kubeflow/kubeflow>`__, called kubeflow mode.
Before starting to use NNI kubeflow mode, you should have a Kubernetes cluster,
either on-premises or `Azure Kubernetes Service(AKS) <https://azure.microsoft.com/en-us/services/kubernetes-service/>`__,
a Ubuntu machine on which `kubeconfig <https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/>`__
is setup to connect to your Kubernetes cluster.
If you are not familiar with Kubernetes, `here <https://kubernetes.io/docs/tutorials/kubernetes-basics/>`__ is a good start.
In kubeflow mode, your trial program will run as Kubeflow job in Kubernetes cluster.

Prerequisite for on-premises Kubernetes Service
-----------------------------------------------

1. A **Kubernetes** cluster using Kubernetes 1.8 or later.
   Follow this `guideline <https://kubernetes.io/docs/setup/>`__ to set up Kubernetes.
2. Download, set up, and deploy **Kubeflow** to your Kubernetes cluster.
   Follow this `guideline <https://www.kubeflow.org/docs/started/getting-started/>`__ to setup Kubeflow.
3. Prepare a **kubeconfig** file, which will be used by NNI to interact with your Kubernetes API server.
   By default, NNI manager will use ``~/.kube/config`` as kubeconfig file's path.
   You can also specify other kubeconfig files by setting the **KUBECONFIG** environment variable.
   Refer this `guideline <https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig>`__
   to learn more about kubeconfig.
4. If your NNI trial job needs GPU resource, you should follow this `guideline <https://github.com/NVIDIA/k8s-device-plugin>`__
   to configure **Nvidia device plugin for Kubernetes**.
5. Prepare a **NFS server** and export a general purpose mount
   (we recommend to map your NFS server path in ``root_squash option``,
   otherwise permission issue may raise when NNI copy files to NFS.
   Refer this `page <https://linux.die.net/man/5/exports>`__ to learn what root_squash option is),
   or **Azure File Storage**.
6. Install **NFS client** on the machine where you install NNI and run nnictl to create experiment.
   Run this command to install NFSv4 client:

.. code-block:: bash

    apt install nfs-common

7. Install **NNI**:

.. code-block:: bash

    python -m pip install nni

Prerequisite for Azure Kubernetes Service
-----------------------------------------

1. NNI support Kubeflow based on Azure Kubernetes Service,
   follow the `guideline <https://azure.microsoft.com/en-us/services/kubernetes-service/>`__ to set up Azure Kubernetes Service.
2. Install `Azure CLI <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest>`__ and **kubectl**.
   Use ``az login`` to set azure account, and connect kubectl client to AKS,
   refer this `guideline <https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough#connect-to-the-cluster>`__.
3. Deploy Kubeflow on Azure Kubernetes Service, follow the `guideline <https://www.kubeflow.org/docs/started/getting-started/>`__.
4. Follow the `guideline <https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?tabs=portal>`__
   to create azure file storage account.
   If you use Azure Kubernetes Service, NNI need Azure Storage Service to store code files and the output files.
5. To access Azure storage service, NNI need the access key of the storage account,
   and NNI use `Azure Key Vault <https://azure.microsoft.com/en-us/services/key-vault/>`__ Service to protect your private key.
   Set up Azure Key Vault Service, add a secret to Key Vault to store the access key of Azure storage account.
   Follow this `guideline <https://docs.microsoft.com/en-us/azure/key-vault/quick-create-cli>`__ to store the access key.

Design
------

.. image:: ../../img/kubeflow_training_design.png
   :target: ../../img/kubeflow_training_design.png
   :alt: 

Kubeflow training service instantiates a Kubernetes rest client to interact with your K8s cluster's API server.

For each trial, we will upload all the files in your local ``trial_code_directory``
together with NNI generated files like parameter.cfg into a storage volumn.
Right now we support two kinds of storage volumes:
`nfs <https://en.wikipedia.org/wiki/Network_File_System>`__
and `azure file storage <https://azure.microsoft.com/en-us/services/storage/files/>`__,
you should configure the storage volumn in experiment config.
After files are prepared, Kubeflow training service will call K8S rest API to create Kubeflow jobs
(`tf-operator <https://github.com/kubeflow/tf-operator>`__ job
or `pytorch-operator <https://github.com/kubeflow/pytorch-operator>`__ job)
in K8S, and mount your storage volume into the job's pod.
Output files of Kubeflow job, like stdout, stderr, trial.log or model files, will also be copied back to the storage volumn.
NNI will show the storage volumn's URL for each trial in web portal, to allow user browse the log files and job's output files.

Supported operator
------------------

NNI only support tf-operator and pytorch-operator of Kubeflow, other operators are not tested.
Users can set operator type in experiment config.
The setting of tf-operator:

.. code-block:: yaml

    config.training_service.operator = 'tf-operator'

The setting of pytorch-operator:

.. code-block:: yaml

    config.training_service.operator = 'pytorch-operator'

If users want to use tf-operator, he could set ``ps`` and ``worker`` in trial config.
If users want to use pytorch-operator, he could set ``master`` and ``worker`` in trial config.

Supported storage type
----------------------

NNI support NFS and Azure Storage to store the code and output files,
users could set storage type in config file and set the corresponding config.

The setting for NFS storage are as follows:

.. code-block:: python

    config.training_service.storage = K8sNfsConfig(
        server = '10.20.30.40', # your NFS server IP
        path = '/mnt/nfs/nni'   # your NFS server export path
    )

If you use Azure storage, you should set ``storage`` in your config as follows:

.. code-block:: python

    config.training_service.storage = K8sAzureStorageConfig(
        azure_account = your_azure_account_name,
        azure_share = your_azure_share_name,
        key_vault_name = your_vault_name,
        key_vault_key = your_secret_name
    )

Run an experiment
-----------------

Use :doc:`PyTorch quickstart </tutorials/hpo_quickstart_pytorch/main>` as an example.
This is a PyTorch job, and use pytorch-operator of Kubeflow.
The experiment config is like:

.. code-block:: python

    from nni.experiment import Experiment, K8sNfsConfig, KubeflowRowConfig

    experiment = Experiment('kubeflow')
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 10
    experiment.config.trial_concurrency = 2

    experiment.config.operator = 'pytorch-operator'
    experiment.config.api_version = 'v1alpha2'

    experiment.config.training_service.storage = K8sNfsConfig()
    experiment.config.training_service.storage.server = '10.20.30.40'
    experiment.config.training_service.storage.path = '/mnt/nfs/nni'

    experiment.config.training_service.worker = KubeflowRowConfig()
    experiment.config.training_service.worker.replicas = 2
    experiment.config.training_service.worker.command = 'python3 model.py'
    experiment.config.training_service.worker.gpuNumber = 1
    experiment.config.training_service.worker.cpuNumber = 1
    experiment.config.training_service.worker.memorySize = '4g'
    experiment.config.training_service.worker.code_directory = '.'

    experiment.config.training_service.master = KubeflowRowConfig()
    experiment.config.training_service.master.replicas = 1
    experiment.config.training_service.master.command = 'python3 model.py'
    experiment.config.training_service.master.gpuNumber = 0
    experiment.config.training_service.master.cpuNumber = 1
    experiment.config.training_service.master.memorySize = '4g'
    experiment.config.training_service.master.code_directory = '.'
    experiment.config.training_service.worker.docker_image = 'msranni/nni:latest'  # default

Once it's ready, run:

.. code-block:: python

    experiment.run(8080)

NNI will create Kubeflow pytorchjob for each trial,
and the job name format is something like ``nni_exp_{experiment_id}_trial_{trial_id}``.
You can see the Kubeflow jobs created by NNI in your Kubernetes dashboard.

Notice: In kubeflow mode, NNIManager will start a rest server and listen on a port which is your NNI web portal's port plus 1.
For example, if your web portal port is ``8080``, the rest server will listen on ``8081``,
to receive metrics from trial job running in Kubernetes.
So you should ``enable 8081`` TCP port in your firewall rule to allow incoming traffic.

Once a trial job is completed, you can go to NNI web portal's overview page (like http://localhost:8080/oview)
to check trials' information.
