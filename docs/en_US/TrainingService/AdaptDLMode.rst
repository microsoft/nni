Run an Experiment on AdaptDL
============================

Now NNI supports running experiment on `AdaptDL <https://github.com/petuum/adaptdl>`__. Before starting to use NNI AdaptDL mode, you should have a Kubernetes cluster, either on-premises or `Azure Kubernetes Service(AKS) <https://azure.microsoft.com/en-us/services/kubernetes-service/>`__\ , a Ubuntu machine on which `kubeconfig <https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/>`__ is setup to connect to your Kubernetes cluster. In AdaptDL mode, your trial program will run as AdaptDL job in Kubernetes cluster.

AdaptDL aims to make distributed deep learning easy and efficient in dynamic-resource environments such as shared clusters and the cloud.

Prerequisite for Kubernetes Service
-----------------------------------


#. A **Kubernetes** cluster using Kubernetes 1.14 or later with storage. Follow this guideline to set up Kubernetes `on Azure <https://azure.microsoft.com/en-us/services/kubernetes-service/>`__\ , or `on-premise <https://kubernetes.io/docs/setup/>`__ with `cephfs <https://kubernetes.io/docs/concepts/storage/storage-classes/#ceph-rbd>`__\ , or `microk8s with storage add-on enabled <https://microk8s.io/docs/addons>`__.
#. Helm install **AdaptDL Scheduler** to your Kubernetes cluster. Follow this `guideline <https://adaptdl.readthedocs.io/en/latest/installation/install-adaptdl.html>`__ to setup AdaptDL scheduler.
#. Prepare a **kubeconfig** file, which will be used by NNI to interact with your Kubernetes API server. By default, NNI manager will use $(HOME)/.kube/config as kubeconfig file's path. You can also specify other kubeconfig files by setting the** KUBECONFIG** environment variable. Refer this `guideline <https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig>`__ to learn more about kubeconfig.
#. If your NNI trial job needs GPU resource, you should follow this `guideline <https://github.com/NVIDIA/k8s-device-plugin>`__ to configure **Nvidia device plugin for Kubernetes**.
#. (Optional) Prepare a **NFS server** and export a general purpose mount as external storage.
#. Install **NNI**\ , follow the install guide `here <../Tutorial/QuickStart.rst>`__.

Verify Prerequisites
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   nnictl --version
   # Expected: <version_number>

.. code-block:: bash

   kubectl version
   # Expected that the kubectl client version matches the server version.

.. code-block:: bash

   kubectl api-versions | grep adaptdl
   # Expected: adaptdl.petuum.com/v1

Run an experiment
-----------------

We have a CIFAR10 example that fully leverages the AdaptDL scheduler under ``examples/trials/cifar10_pytorch`` folder. (\ ``main_adl.py`` and ``config_adl.yaml``\ )

Here is a template configuration specification to use AdaptDL as a training service.

.. code-block:: yaml

   authorName: default
   experimentName: minimal_adl

   trainingServicePlatform: adl
   nniManagerIp: 10.1.10.11
   logCollection: http

   tuner:
     builtinTunerName: GridSearch
   searchSpacePath: search_space.json

   trialConcurrency: 2
   maxTrialNum: 2

   trial:
     adaptive: false # optional.
     image: <image_tag>
     imagePullSecrets:  # optional
       - name: stagingsecret
     codeDir: .
     command: python main.py
     gpuNum: 1
     cpuNum: 1  # optional
     memorySize: 8Gi  # optional
     nfs: # optional
       server: 10.20.41.55
       path: /
       containerMountPath: /nfs
     checkpoint: # optional
       storageClass: dfs
       storageSize: 1Gi

Those configs not mentioned below, are following the
`default specs defined in the NNI doc </Tutorial/ExperimentConfig.html#configuration-spec>`__.


* **trainingServicePlatform**\ : Choose ``adl`` to use the Kubernetes cluster with AdaptDL scheduler.
* **nniManagerIp**\ : *Required* to get the correct info and metrics back from the cluster, for ``adl`` training service.
  IP address of the machine with NNI manager (NNICTL) that launches NNI experiment.
* **logCollection**\ : *Recommended* to set as ``http``. It will collect the trial logs on cluster back to your machine via http.
* **tuner**\ : It supports the Tuun tuner and all NNI built-in tuners (only except for the checkpoint feature of the NNI PBT tuners).
* **trial**\ : It defines the specs of an ``adl`` trial.

  * **namespace**\: (*Optional*\ ) Kubernetes namespace to launch the trials. Default to ``default`` namespace.
  * **adaptive**\ : (*Optional*\ ) Boolean for AdaptDL trainer. While ``true``\ , it the job is preemptible and adaptive.
  * **image**\ : Docker image for the trial
  * **imagePullSecret**\ : (*Optional*\ ) If you are using a private registry,
    you need to provide the secret to successfully pull the image.
  * **codeDir**\ : the working directory of the container. ``.`` means the default working directory defined by the image.
  * **command**\ : the bash command to start the trial
  * **gpuNum**\ : the number of GPUs requested for this trial. It must be non-negative integer.
  * **cpuNum**\ : (*Optional*\ ) the number of CPUs requested for this trial.  It must be non-negative integer.
  * **memorySize**\ : (*Optional*\ ) the size of memory requested for this trial. It must follow the Kubernetes
    `default format <https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory>`__.
  * **nfs**\ : (*Optional*\ ) mounting external storage. For more information about using NFS please check the below paragraph.
  * **checkpoint** (*Optional*\ ) storage settings for model checkpoints.

    * **storageClass**\ : check `Kubernetes storage documentation <https://kubernetes.io/docs/concepts/storage/storage-classes/>`__ for how to use the appropriate ``storageClass``.
    * **storageSize**\ : this value should be large enough to fit your model's checkpoints, or it could cause "disk quota exceeded" error.

NFS Storage
^^^^^^^^^^^

As you may have noticed in the above configuration spec,
an *optional* section is available to configure NFS external storage. It is optional when no external storage is required, when for example an docker image is sufficient with codes and data inside.

Note that ``adl`` training service does NOT help mount an NFS to the local dev machine, so that one can manually mount it to local, manage the filesystem, copy the data or code etc.
The ``adl`` training service can then mount it to the kubernetes for every trials, with the proper configurations:


* **server**\ : NFS server address, e.g. IP address or domain
* **path**\ : NFS server export path, i.e. the absolute path in NFS that can be mounted to trials
* **containerMountPath**\ : In container absolute path to mount the NFS **path** above,
  so that every trial will have the access to the NFS.
  In the trial containers, you can access the NFS with this path.

Use cases:


* If your training trials depend on a dataset of large size, you may want to download it first onto the NFS first,
  and mount it so that it can be shared across multiple trials.
* The storage for containers are ephemeral and the trial containers will be deleted after a trial's lifecycle is over.
  So if you want to export your trained models,
  you may mount the NFS to the trial to persist and export your trained models.

In short, it is not limited how a trial wants to read from or write on the NFS storage, so you may use it flexibly as per your needs.

Monitor via Log Stream
----------------------

Follow the log streaming of a certain trial:

.. code-block:: bash

   nnictl log trial --trial_id=<trial_id>

.. code-block:: bash

   nnictl log trial <experiment_id> --trial_id=<trial_id>

Note that *after* a trial has done and its pod has been deleted,
no logs can be retrieved then via this command.
However you may still be able to access the past trial logs
according to the following approach.

Monitor via TensorBoard
-----------------------

In the context of NNI, an experiment has multiple trials.
For easy comparison across trials for a model tuning process,
we support TensorBoard integration. Here one experiment has
an independent TensorBoard logging directory thus dashboard.

You can only use the TensorBoard while the monitored experiment is running.
In other words, it is not supported to monitor stopped experiments.

In the trial container you may have access to two environment variables:


* ``ADAPTDL_TENSORBOARD_LOGDIR``\ : the TensorBoard logging directory for the current experiment,
* ``NNI_TRIAL_JOB_ID``\ : the ``trial`` job id for the current trial.

It is recommended for to have them joined as the directory for trial,
for example in Python:

.. code-block:: python

   import os
   tensorboard_logdir = os.path.join(
       os.getenv("ADAPTDL_TENSORBOARD_LOGDIR"),
       os.getenv("NNI_TRIAL_JOB_ID")
   )

If an experiment is stopped, the data logged here
(defined by *the above envs* for monitoring with the following commands)
will be lost. To persist the logged data, you can use the external storage (e.g. to mount an NFS)
to export it and view the TensorBoard locally.

With the above setting, you can monitor the experiment easily
via TensorBoard by

.. code-block:: bash

   nnictl tensorboard start

If having multiple experiment running at the same time, you may use

.. code-block:: bash

   nnictl tensorboard start <experiment_id>

It will provide you the web url to access the tensorboard.

Note that you have the flexibility to set up the local ``--port``
for the TensorBoard.
