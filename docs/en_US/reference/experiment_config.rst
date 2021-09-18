===========================
Experiment Config Reference
===========================

A config file is needed when creating an experiment. This document describes the rules to write a config file and provides some examples.

.. Note::

    1. This document lists field names with ``camelCase``. If users use these fields in the pythonic way with NNI Python APIs (e.g., ``nni.experiment``), the field names should be converted to ``snake_case``.

    2. In this document, the type of fields are formatted as `Python type hint <https://docs.python.org/3.10/library/typing.html>`_. Therefore JSON objects are called `dict` and arrays are called `list`.

    .. _path: 

    3. Some fields take a path to a file or directory. Unless otherwise noted, both absolute path and relative path are supported, and ``~`` will be expanded to the home directory.

       - When written in the YAML file, relative paths are relative to the directory containing that file.
       - When assigned in Python code, relative paths are relative to the current working directory.
       - All relative paths are converted to absolute when loading YAML file into Python class, and when saving Python class to YAML file.

    4. Setting a field to ``None`` or ``null`` is equivalent to not setting the field.

.. contents:: Contents
   :local:
   :depth: 3
 

Examples
========

Local Mode
^^^^^^^^^^

.. code-block:: yaml

    experimentName: MNIST
    searchSpaceFile: search_space.json
    trialCommand: python mnist.py
    trialCodeDirectory: .
    trialGpuNumber: 1
    trialConcurrency: 2
    maxExperimentDuration: 24h
    maxTrialNumber: 100
    tuner:
      name: TPE
      classArgs:
        optimize_mode: maximize
    trainingService:
      platform: local
      useActiveGpu: True

Local Mode (Inline Search Space)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    searchSpace:
      batch_size:
        _type: choice
        _value: [16, 32, 64]
      learning_rate:
        _type: loguniform
        _value: [0.0001, 0.1]
    trialCommand: python mnist.py
    trialGpuNumber: 1
    trialConcurrency: 2
    tuner:
      name: TPE
      classArgs:
        optimize_mode: maximize
    trainingService:
      platform: local
      useActiveGpu: True

Remote Mode
^^^^^^^^^^^

.. code-block:: yaml

    experimentName: MNIST
    searchSpaceFile: search_space.json
    trialCommand: python mnist.py
    trialCodeDirectory: .
    trialGpuNumber: 1
    trialConcurrency: 2
    maxExperimentDuration: 24h
    maxTrialNumber: 100
    tuner:
      name: TPE
      classArgs:
        optimize_mode: maximize
    trainingService:
      platform: remote
      machineList:
        - host: 11.22.33.44
          user: alice
          password: xxxxx
        - host: my.domain.com
          user: bob
          sshKeyFile: ~/.ssh/id_rsa

Reference
=========

ExperimentConfig
^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description
    
    * - experimentName
      - ``Optional[str]``
      - Mnemonic name of the experiment, which will be shown in WebUI and nnictl.

    * - searchSpaceFile
      - ``Optional[str]``
      - Path_ to the JSON file containing the search space.
        Search space format is determined by tuner. The common format for built-in tuners is documented  `here <../Tutorial/SearchSpaceSpec.rst>`__.
        Mutually exclusive to ``searchSpace``.

    * - searchSpace
      - ``Optional[JSON]``
      - Search space object.
        The format is determined by tuner. Common format for built-in tuners is documented `here <../Tutorial/SearchSpaceSpec.rst>`__.
        Note that ``None`` means "no such field" so empty search space should be written as ``{}``.
        Mutually exclusive to ``searchSpaceFile``.

    * - trialCommand
      - ``str``
      - Command to launch trial.
        The command will be executed in bash on Linux and macOS, and in PowerShell on Windows.
        Note that using ``python3`` on Linux and macOS, and using ``python`` on Windows.

    * - trialCodeDirectory
      - ``str``
      - `Path`_ to the directory containing trial source files.
        default: ``"."``.
        All files in this directory will be sent to the training machine, unless in the ``.nniignore`` file.
        (See :ref:`nniignore <nniignore>` for details.)

    * - trialConcurrency
      - ``int``
      - Specify how many trials should be run concurrently.
        The real concurrency also depends on hardware resources and may be less than this value.

    * - trialGpuNumber
      - ``Optional[int]``
      - This field might have slightly different meanings for various training services,
        especially when set to ``0`` or ``None``.
        See `training service's document <../training_services.rst>`__ for details.

        In local mode, setting the field to ``0`` will prevent trials from accessing GPU (by empty ``CUDA_VISIBLE_DEVICES``).
        And when set to ``None``, trials will be created and scheduled as if they did not use GPU,
        but they can still use all GPU resources if they want.

    * - maxExperimentDuration
      - ``Optional[str]``
      - Limit the duration of this experiment if specified.
        format: ``number + s|m|h|d``
        examples: ``"10m"``, ``"0.5h"``
        When time runs out, the experiment will stop creating trials but continue to serve WebUI.

    * - maxTrialNumber
      - ``Optional[int]``
      - Limit the number of trials to create if specified.
        When the budget runs out, the experiment will stop creating trials but continue to serve WebUI.

    * - maxTrialDuration
      - ``Optional[str]``
      - Limit the duration of trial job if specified.
        format: ``number + s|m|h|d``
        examples: ``"10m"``, ``"0.5h"``
        When time runs out, the current trial job will stop.

    * - nniManagerIp
      - ``Optional[str]``
      - IP of the current machine, used by training machines to access NNI manager. Not used in local mode.
        If not specified, IPv4 address of ``eth0`` will be used.
        Except for the local mode, it is highly recommended to set this field manually.

    * - useAnnotation
      - ``bool``
      - Enable `annotation <../Tutorial/AnnotationSpec.rst>`__.
        default: ``False``.
        When using annotation, ``searchSpace`` and ``searchSpaceFile`` should not be specified manually.

    * - debug
      - ``bool``
      - Enable debug mode.
        default: ``False``
        When enabled, logging will be more verbose and some internal validation will be loosened.

    * - logLevel
      - ``Optional[str]``
      - Set log level of the whole system.
        values: ``"trace"``, ``"debug"``, ``"info"``, ``"warning"``, ``"error"``, ``"fatal"``
        Defaults to "info" or "debug", depending on ``debug`` option. When debug mode is enabled, Loglevel is set to "debug", otherwise, Loglevel is set to "info".
        Most modules of NNI will be affected by this value, including NNI manager, tuner, training service, etc.
        The exception is trial, whose logging level is directly managed by trial code.
        For Python modules, "trace" acts as logging level 0 and "fatal" acts as ``logging.CRITICAL``.

    * - experimentWorkingDirectory
      - ``Optional[str]``
      - Specify the :ref:`directory <path>` to place log, checkpoint, metadata, and other run-time stuff.
        By default uses ``~/nni-experiments``.
        NNI will create a subdirectory named by experiment ID, so it is safe to use the same directory for multiple experiments.

    * - tunerGpuIndices
      - ``Optional[list[int] | str | int]``
      - Limit the GPUs visible to tuner, assessor, and advisor.
        This will be the ``CUDA_VISIBLE_DEVICES`` environment variable of tuner process.
        Because tuner, assessor, and advisor run in the same process, this option will affect them all.

    * - tuner
      - ``Optional[AlgorithmConfig]``
      - Specify the tuner.
        The built-in tuners can be found `here <../builtin_tuner.rst>`__ and you can follow `this tutorial <../Tuner/CustomizeTuner.rst>`__ to customize a new tuner.

    * - assessor
      - ``Optional[AlgorithmConfig]``
      - Specify the assessor.
        The built-in assessors can be found `here <../builtin_assessor.rst>`__ and you can follow `this tutorial <../Assessor/CustomizeAssessor.rst>`__ to customize a new assessor.

    * - advisor
      - ``Optional[AlgorithmConfig]``
      - Specify the advisor.
        NNI provides two built-in advisors: `BOHB <../Tuner/BohbAdvisor.rst>`__ and `Hyperband <../Tuner/HyperbandAdvisor.rst>`__, and you can follow `this tutorial <../Tuner/CustomizeAdvisor.rst>`__ to customize a new advisor.

    * - trainingService
      - ``TrainingServiceConfig``
      - Specify the `training service <../TrainingService/Overview.rst>`__.

    * - sharedStorage
      - ``Optional[SharedStorageConfig]``
      - Configure the shared storage, detailed usage can be found `here <../Tutorial/HowToUseSharedStorage.rst>`__.

AlgorithmConfig
^^^^^^^^^^^^^^^

``AlgorithmConfig`` describes a tuner / assessor / advisor algorithm.

For customized algorithms, there are two ways to describe them:

  1. `Register the algorithm <../Tutorial/InstallCustomizedAlgos.rst>`__ to use it like built-in. (preferred)

  2. Specify code directory and class name directly.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description
    
    * - name
      - ``Optional[str]``
      - Name of the built-in or registered algorithm.
        ``str`` for the built-in and registered algorithm, ``None`` for other customized algorithms.

    * - className
      - ``Optional[str]``
      - Qualified class name of not registered customized algorithm.
        ``None`` for the built-in and registered algorithm, ``str`` for other customized algorithms.
        example: ``"my_tuner.MyTuner"``

    * - codeDirectory
      - ``Optional[str]``
      - `Path`_ to the directory containing the customized algorithm class.
        ``None`` for the built-in and registered algorithm, ``str`` for other customized algorithms.

    * - classArgs
      - ``Optional[dict[str, Any]]``
      - Keyword arguments passed to algorithm class' constructor.
        See algorithm's document for supported value.

TrainingServiceConfig
^^^^^^^^^^^^^^^^^^^^^

One of the following:

- `LocalConfig`_
- `RemoteConfig`_
- `OpenpaiConfig`_
- `AmlConfig`_
- `DlcConfig`_
- `HybridConfig`_

For `Kubeflow <../TrainingService/KubeflowMode.rst>`_, `FrameworkController <../TrainingService/FrameworkControllerMode.rst>`_, and `AdaptDL <../TrainingService/AdaptDLMode.rst>`_ training platforms, it is suggested to use `v1 config schema <../Tutorial/ExperimentConfig.rst>`_ for now.

LocalConfig
-----------

Detailed usage can be found `here <../TrainingService/LocalMode.rst>`__.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - platform
      - ``"local"``
      -
    
    * - useActiveGpu
      - ``Optional[bool]``
      - Specify whether NNI should submit trials to GPUs occupied by other tasks.
        Must be set when ``trialGpuNumber`` greater than zero.
        Following processes can make GPU "active":

          - non-NNI CUDA programs
          - graphical desktop
          - trials submitted by other NNI instances, if you have more than one NNI experiments running at same time
          - other users' CUDA programs, if you are using a shared server
          
        If you are using a graphical OS like Windows 10 or Ubuntu desktop, set this field to ``True``, otherwise, the GUI will prevent NNI from launching any trial.
        When you create multiple NNI experiments and ``useActiveGpu`` is set to ``True``, they will submit multiple trials to the same GPU(s) simultaneously.

    * - maxTrialNumberPerGpu
      - ``int``
      - Specify how many trials can share one GPU.
        default: ``1``

    * - gpuIndices
      - ``Optional[list[int] | str | int]``
      - Limit the GPUs visible to trial processes.
        If ``trialGpuNumber`` is less than the length of this value, only a subset will be visible to each trial.
        This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.

RemoteConfig
------------

Detailed usage can be found `here <../TrainingService/RemoteMachineMode.rst>`__.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - platform
      - ``"remote"``
      -

    * - machineList
      - ``List[RemoteMachineConfig]``
      - List of training machines.

    * - reuseMode
      - ``bool``
      - Enable `reuse mode <../TrainingService/Overview.rst#training-service-under-reuse-mode>`__.

RemoteMachineConfig
"""""""""""""""""""

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - host
      - ``str``
      - IP or hostname (domain name) of the machine.

    * - port
      - ``int``
      - SSH service port.
        default: ``22``

    * - user
      - ``str``
      - Login user name.

    * - password
      - ``Optional[str]``
      - If not specified, ``sshKeyFile`` will be used instead.
    
    * - sshKeyFile
      - ``Optional[str]``
      - `Path`_ to ``sshKeyFile`` (identity file).
        Only used when ``password`` is not specified.

    * - sshPassphrase
      - ``Optional[str]``
      - Passphrase of SSH identity file.

    * - useActiveGpu
      - ``bool``
      - Specify whether NNI should submit trials to GPUs occupied by other tasks.
        default: ``False``
        Must be set when ``trialGpuNumber`` greater than zero.
        Following processes can make GPU "active":

          - non-NNI CUDA programs
          - graphical desktop
          - trials submitted by other NNI instances, if you have more than one NNI experiments running at same time
          - other users' CUDA programs, if you are using a shared server
  
        If your remote machine is a graphical OS like Ubuntu desktop, set this field to ``True``, otherwise, the GUI will prevent NNI from launching any trial.
        When you create multiple NNI experiments and ``useActiveGpu`` is set to ``True``, they will submit multiple trials to the same GPU(s) simultaneously.

    * - maxTrialNumberPerGpu
      - ``int``
      - Specify how many trials can share one GPU.
        default: ``1``

    * - gpuIndices
      - ``Optional[list[int] | str | int]``
      - Limit the GPUs visible to trial processes.
        If ``trialGpuNumber`` is less than the length of this value, only a subset will be visible to each trial.
        This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.

    * - pythonPath
      - ``Optional[str]``
      - Specify a Python environment.
        This path will be inserted at the front of PATH. Here are some examples: 

          - (linux) pythonPath: ``/opt/python3.7/bin``
          - (windows) pythonPath: ``C:/Python37``

        If you are working on Anaconda, there is some difference. On Windows, you also have to add ``../script`` and ``../Library/bin`` separated by ``;``. Examples are as below:

          - (linux anaconda) pythonPath: ``/home/yourname/anaconda3/envs/myenv/bin/``
          - (windows anaconda) pythonPath: ``C:/Users/yourname/.conda/envs/myenv;C:/Users/yourname/.conda/envs/myenv/Scripts;C:/Users/yourname/.conda/envs/myenv/Library/bin``

        This is useful if preparing steps vary for different machines.

OpenpaiConfig
-------------

Detailed usage can be found `here <../TrainingService/PaiMode.rst>`__.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - platform
      - ``"openpai"``
      -
    
    * - host
      - ``str``
      - Hostname of OpenPAI service.
        This may include ``https://`` or ``http://`` prefix.
        HTTPS will be used by default.

    * - username
      - ``str``
      - OpenPAI user name.

    * - token
      - ``str``
      - OpenPAI user token.
        This can be found in your OpenPAI user settings page.

    * - trialCpuNumber
      - ``int``
      - Specify the CPU number of each trial to be used in OpenPAI container.

    * - trialMemorySize
      - ``str``
      - Specify the memory size of each trial to be used in OpenPAI container.
        format: ``number + tb|gb|mb|kb``.
        examples: ``"8gb"``, ``"8192mb"``.

    * - storageConfigName
      - ``str``
      - Specify the storage name used in OpenPAI.

    * - dockerImage
      - ``str``
      - Name and tag of docker image to run the trials.
        default: ``"msranni/nni:latest"``.

    * - localStorageMountPoint
      - ``str``
      - :ref:`Mount point <path>` of storage service (typically NFS) on the local machine.

    * - containerStorageMountPoint
      - ``str``
      - Mount point of storage service (typically NFS) in docker container.
        This must be an absolute path.

    * - reuseMode
      - ``bool``
      - Enable `reuse mode <../TrainingService/Overview.rst#training-service-under-reuse-mode>`__.
        default: ``False``.

    * - openpaiConfig
      - ``Optional[JSON]``
      - Embedded OpenPAI config file.

    * - openpaiConfigFile
      - ``Optional[str]``
      - `Path`_ to OpenPAI config file.
        An example can be found `here <https://github.com/microsoft/pai/blob/master/docs/manual/cluster-user/examples/hello-world-job.yaml>`__.

AmlConfig
---------

Detailed usage can be found `here <../TrainingService/AMLMode.rst>`__.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - platform
      - ``"aml"``
      -
    
    * - dockerImage
      - ``str``
      - Name and tag of docker image to run the trials.
        default: ``"msranni/nni:latest"``

    * - subscriptionId
      - ``str``
      - Azure subscription ID.

    * - resourceGroup
      - ``str``
      - Azure resource group name.

    * - workspaceName
      - ``str``
      - Azure workspace name.

    * - computeTarget
      - ``str``
      - AML compute cluster name.

DlcConfig
---------

Detailed usage can be found `here <../TrainingService/DlcMode.rst>`__.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - platform
      - ``"dlc"``
      -
    
    * - type
      - ``str``
      - Job spec type.
        default: ``"worker"``.

    * - image
      - ``str``
      - Name and tag of docker image to run the trials.

    * - jobType
      - ``str``
      - PAI-DLC training job type, ``"TFJob"`` or ``"PyTorchJob"``.

    * - podCount
      - ``str``
      - Pod count to run a single training job.

    * - ecsSpec
      - ``str``
      - Training server config spec string.

    * - region
      - ``str``
      - The region where PAI-DLC public-cluster locates.

    * - nasDataSourceId
      - ``str``
      - The NAS datasource id configurated in PAI-DLC side.

    * - accessKeyId
      - ``str``
      - The accessKeyId of your cloud account.

    * - accessKeySecret
      - ``str``
      - The accessKeySecret of your cloud account.

    * - localStorageMountPoint
      - ``str``
      - The mount point of the NAS on PAI-DSW server, default is /home/admin/workspace/.

    * - containerStorageMountPoint
      - ``str``
      - The mount point of the NAS on PAI-DLC side, default is /root/data/.

HybridConfig
------------

Currently only support `LocalConfig`_, `RemoteConfig`_, `OpenpaiConfig`_ and `AmlConfig`_ . Detailed usage can be found `here <../TrainingService/HybridMode.rst>`__.

SharedStorageConfig
^^^^^^^^^^^^^^^^^^^

Detailed usage can be found `here <../Tutorial/HowToUseSharedStorage.rst>`__.

nfsConfig
---------

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - storageType
      - ``"NFS"``
      -

    * - localMountPoint
      - ``str``
      - The path that the storage has been or will be mounted in the local machine.
        If the path does not exist, it will be created automatically. Recommended to use an absolute path, i.e. ``/tmp/nni-shared-storage``.

    * - remoteMountPoint
      - ``str``
      - The path that the storage will be mounted in the remote machine.
        If the path does not exist, it will be created automatically. Recommended to use a relative path. i.e. ``./nni-shared-storage``.

    * - localMounted
      - ``str``
      - Specify the object and status to mount the shared storage.
        values: ``"usermount"``, ``"nnimount"``, ``"nomount"``
        ``usermount`` means the user has already mounted this storage on localMountPoint. ``nnimount`` means NNI will try to mount this storage on localMountPoint. ``nomount`` means storage will not mount in the local machine, will support partial storages in the future.

    * - nfsServer
      - ``str``
      - NFS server host.

    * - exportedDirectory
      - ``str``
      - Exported directory of NFS server, detailed `here <https://www.ibm.com/docs/en/aix/7.2?topic=system-nfs-exporting-mounting>`_.

azureBlobConfig
---------------

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - storageType
      - ``"AzureBlob"``
      -

    * - localMountPoint
      - ``str``
      - The path that the storage has been or will be mounted in the local machine.
        If the path does not exist, it will be created automatically. Recommended to use an absolute path, i.e. ``/tmp/nni-shared-storage``.

    * - remoteMountPoint
      - ``str``
      - The path that the storage will be mounted in the remote machine.
        If the path does not exist, it will be created automatically. Recommended to use a relative path. i.e. ``./nni-shared-storage``.
        Note that the directory must be empty when using AzureBlob.

    * - localMounted
      - ``str``
      - Specify the object and status to mount the shared storage.
        values: ``"usermount"``, ``"nnimount"``, ``"nomount"``.
        ``usermount`` means the user has already mounted this storage on localMountPoint. ``nnimount`` means NNI will try to mount this storage on localMountPoint. ``nomount`` means storage will not mount in the local machine, will support partial storages in the future.

    * - storageAccountName
      - ``str``
      - Azure storage account name.

    * - storageAccountKey
      - ``Optional[str]``
      - Azure storage account key.

    * - containerName
      - ``str``
      - AzureBlob container name.
