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

experimentName
--------------

Mnemonic name of the experiment, which will be shown in WebUI and nnictl.

type: ``Optional[str]``


searchSpaceFile
---------------

Path_ to the JSON file containing the search space.

type: ``Optional[str]``

Search space format is determined by tuner. The common format for built-in tuners is documented  `here <../Tutorial/SearchSpaceSpec.rst>`__.

Mutually exclusive to `searchSpace`_.


searchSpace
-----------

Search space object.

type: ``Optional[JSON]``

The format is determined by tuner. Common format for built-in tuners is documented `here <../Tutorial/SearchSpaceSpec.rst>`__.

Note that ``None`` means "no such field" so empty search space should be written as ``{}``.

Mutually exclusive to `searchSpaceFile`_.


trialCommand
------------

Command to launch trial.

type: ``str``

The command will be executed in bash on Linux and macOS, and in PowerShell on Windows.

Note that using ``python3`` on Linux and macOS, and using ``python`` on Windows.


trialCodeDirectory
------------------

`Path`_ to the directory containing trial source files.

type: ``str``

default: ``"."``

All files in this directory will be sent to the training machine, unless in the ``.nniignore`` file.
(See :ref:`nniignore <nniignore>` for details.)


trialConcurrency
----------------

Specify how many trials should be run concurrently.

type: ``int``

The real concurrency also depends on hardware resources and may be less than this value.


trialGpuNumber
--------------

Number of GPUs used by each trial.

type: ``Optional[int]``

This field might have slightly different meanings for various training services,
especially when set to ``0`` or ``None``.
See `training service's document <../training_services.rst>`__ for details.

In local mode, setting the field to ``0`` will prevent trials from accessing GPU (by empty ``CUDA_VISIBLE_DEVICES``).
And when set to ``None``, trials will be created and scheduled as if they did not use GPU,
but they can still use all GPU resources if they want.


maxExperimentDuration
---------------------

Limit the duration of this experiment if specified.

type: ``Optional[str]``

format: ``number + s|m|h|d``

examples: ``"10m"``, ``"0.5h"``

When time runs out, the experiment will stop creating trials but continue to serve WebUI.


maxTrialNumber
--------------

Limit the number of trials to create if specified.

type: ``Optional[int]``

When the budget runs out, the experiment will stop creating trials but continue to serve WebUI.


nniManagerIp
------------

IP of the current machine, used by training machines to access NNI manager. Not used in local mode.

type: ``Optional[str]``

If not specified, IPv4 address of ``eth0`` will be used.

Except for the local mode, it is highly recommended to set this field manually.


useAnnotation
-------------

Enable `annotation <../Tutorial/AnnotationSpec.rst>`__.

type: ``bool``

default: ``False``

When using annotation, `searchSpace`_ and `searchSpaceFile`_ should not be specified manually.


debug
-----

Enable debug mode.

type: ``bool``

default: ``False``

When enabled, logging will be more verbose and some internal validation will be loosened.


logLevel
--------

Set log level of the whole system.

type: ``Optional[str]``

values: ``"trace"``, ``"debug"``, ``"info"``, ``"warning"``, ``"error"``, ``"fatal"``

Defaults to "info" or "debug", depending on `debug`_ option. When debug mode is enabled, Loglevel is set to "debug", otherwise, Loglevel is set to "info".

Most modules of NNI will be affected by this value, including NNI manager, tuner, training service, etc.

The exception is trial, whose logging level is directly managed by trial code.

For Python modules, "trace" acts as logging level 0 and "fatal" acts as ``logging.CRITICAL``.


experimentWorkingDirectory
--------------------------

Specify the :ref:`directory <path>` to place log, checkpoint, metadata, and other run-time stuff.

type: ``Optional[str]``

By default uses ``~/nni-experiments``.

NNI will create a subdirectory named by experiment ID, so it is safe to use the same directory for multiple experiments.


tunerGpuIndices
---------------

Limit the GPUs visible to tuner, assessor, and advisor.

type: ``Optional[list[int] | str | int]``

This will be the ``CUDA_VISIBLE_DEVICES`` environment variable of tuner process.

Because tuner, assessor, and advisor run in the same process, this option will affect them all.


tuner
-----

Specify the tuner. 

type: Optional `AlgorithmConfig`_

The built-in tuners can be found `here <../builtin_tuner.rst>`__ and you can follow `this tutorial <../Tuner/CustomizeTuner.rst>`__ to customize a new tuner.


assessor
--------

Specify the assessor. 

type: Optional `AlgorithmConfig`_

The built-in assessors can be found `here <../builtin_assessor.rst>`__ and you can follow `this tutorial <../Assessor/CustomizeAssessor.rst>`__ to customize a new assessor.


advisor
-------

Specify the advisor. 

type: Optional `AlgorithmConfig`_

NNI provides two built-in advisors: `BOHB <../Tuner/BohbAdvisor.rst>`__ and `Hyperband <../Tuner/HyperbandAdvisor.rst>`__, and you can follow `this tutorial <../Tuner/CustomizeAdvisor.rst>`__ to customize a new advisor.


trainingService
---------------

Specify the `training service <../TrainingService/Overview.rst>`__.

type: `TrainingServiceConfig`_


sharedStorage
-------------

Configure the shared storage, detailed usage can be found `here <../Tutorial/HowToUseSharedStorage.rst>`__.

type: Optional `SharedStorageConfig`_


AlgorithmConfig
^^^^^^^^^^^^^^^

``AlgorithmConfig`` describes a tuner / assessor / advisor algorithm.

For customized algorithms, there are two ways to describe them:

  1. `Register the algorithm <../Tutorial/InstallCustomizedAlgos.rst>`__ to use it like built-in. (preferred)

  2. Specify code directory and class name directly.


name
----

Name of the built-in or registered algorithm.

type: ``str`` for the built-in and registered algorithm, ``None`` for other customized algorithms.


className
---------

Qualified class name of not registered customized algorithm.

type: ``None`` for the built-in and registered algorithm, ``str`` for other customized algorithms.

example: ``"my_tuner.MyTuner"``


codeDirectory
-------------

`Path`_ to the directory containing the customized algorithm class.

type: ``None`` for the built-in and registered algorithm, ``str`` for other customized algorithms.


classArgs
---------

Keyword arguments passed to algorithm class' constructor.

type: ``Optional[dict[str, Any]]``

See algorithm's document for supported value.


TrainingServiceConfig
^^^^^^^^^^^^^^^^^^^^^

One of the following:

- `LocalConfig`_
- `RemoteConfig`_
- :ref:`OpenpaiConfig <openpai-class>`
- `AmlConfig`_
- `HybridConfig`_

For `Kubeflow <../TrainingService/KubeflowMode.rst>`_, `FrameworkController <../TrainingService/FrameworkControllerMode.rst>`_, and `AdaptDL <../TrainingService/AdaptDLMode.rst>`_ training platforms, it is suggested to use `v1 config schema <../Tutorial/ExperimentConfig.rst>`_ for now.


LocalConfig
-----------

Detailed usage can be found `here <../TrainingService/LocalMode.rst>`__.

platform
""""""""

Constant string ``"local"``.


useActiveGpu
""""""""""""

Specify whether NNI should submit trials to GPUs occupied by other tasks.

type: ``Optional[bool]``

Must be set when `trialGpuNumber`_ greater than zero.

Following processes can make GPU "active":

  - non-NNI CUDA programs
  - graphical desktop
  - trials submitted by other NNI instances, if you have more than one NNI experiments running at same time
  - other users' CUDA programs, if you are using a shared server
  
If you are using a graphical OS like Windows 10 or Ubuntu desktop, set this field to ``True``, otherwise, the GUI will prevent NNI from launching any trial.

When you create multiple NNI experiments and ``useActiveGpu`` is set to ``True``, they will submit multiple trials to the same GPU(s) simultaneously.


maxTrialNumberPerGpu
""""""""""""""""""""

Specify how many trials can share one GPU.

type: ``int``

default: ``1``


gpuIndices
""""""""""

Limit the GPUs visible to trial processes.

type: ``Optional[list[int] | str | int]``

If `trialGpuNumber`_ is less than the length of this value, only a subset will be visible to each trial.

This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.


RemoteConfig
------------

Detailed usage can be found `here <../TrainingService/RemoteMachineMode.rst>`__.

platform
""""""""

Constant string ``"remote"``.


machineList
"""""""""""

List of training machines.

type: list of `RemoteMachineConfig`_


reuseMode
"""""""""

Enable `reuse mode <../TrainingService/Overview.rst#training-service-under-reuse-mode>`__.

type: ``bool``


RemoteMachineConfig
"""""""""""""""""""

host
****

IP or hostname (domain name) of the machine.

type: ``str``


port
****

SSH service port.

type: ``int``

default: ``22``


user
****

Login user name.

type: ``str``


password
********

Login password.

type: ``Optional[str]``

If not specified, `sshKeyFile`_ will be used instead.


sshKeyFile
**********

`Path`_ to sshKeyFile (identity file).

type: ``Optional[str]``

Only used when `password`_ is not specified.


sshPassphrase
*************

Passphrase of SSH identity file.

type: ``Optional[str]``


useActiveGpu
************

Specify whether NNI should submit trials to GPUs occupied by other tasks.

type: ``bool``

default: ``False``

Must be set when `trialGpuNumber`_ greater than zero.

Following processes can make GPU "active":

  - non-NNI CUDA programs
  - graphical desktop
  - trials submitted by other NNI instances, if you have more than one NNI experiments running at same time
  - other users' CUDA programs, if you are using a shared server
  
If your remote machine is a graphical OS like Ubuntu desktop, set this field to ``True``, otherwise, the GUI will prevent NNI from launching any trial.

When you create multiple NNI experiments and ``useActiveGpu`` is set to ``True``, they will submit multiple trials to the same GPU(s) simultaneously.


maxTrialNumberPerGpu
********************

Specify how many trials can share one GPU.

type: ``int``

default: ``1``


gpuIndices
**********

Limit the GPUs visible to trial processes.

type: ``Optional[list[int] | str | int]``

If `trialGpuNumber`_ is less than the length of this value, only a subset will be visible to each trial.

This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.


pythonPath
**********

Specify a Python environment.

type: ``Optional[str]``

This path will be inserted at the front of PATH. Here are some examples: 

    - (linux) pythonPath: ``/opt/python3.7/bin``
    - (windows) pythonPath: ``C:/Python37``

If you are working on Anaconda, there is some difference. On Windows, you also have to add ``../script`` and ``../Library/bin`` separated by ``;``. Examples are as below:

    - (linux anaconda) pythonPath: ``/home/yourname/anaconda3/envs/myenv/bin/``
    - (windows anaconda) pythonPath: ``C:/Users/yourname/.conda/envs/myenv;C:/Users/yourname/.conda/envs/myenv/Scripts;C:/Users/yourname/.conda/envs/myenv/Library/bin``

This is useful if preparing steps vary for different machines.

.. _openpai-class:

OpenpaiConfig
-------------

Detailed usage can be found `here <../TrainingService/PaiMode.rst>`__.

platform
""""""""

Constant string ``"openpai"``.


host
""""

Hostname of OpenPAI service.

type: ``str``

This may include ``https://`` or ``http://`` prefix.

HTTPS will be used by default.


username
""""""""

OpenPAI user name.

type: ``str``


token
"""""

OpenPAI user token.

type: ``str``

This can be found in your OpenPAI user settings page.


trialCpuNumber
""""""""""""""

Specify the CPU number of each trial to be used in OpenPAI container.

type: ``int``


trialMemorySize
"""""""""""""""

Specify the memory size of each trial to be used in OpenPAI container.

type: ``str``

format: ``number + tb|gb|mb|kb``

examples: ``"8gb"``, ``"8192mb"``


storageConfigName
"""""""""""""""""

Specify the storage name used in OpenPAI.

type: ``str``


dockerImage
"""""""""""

Name and tag of docker image to run the trials.

type: ``str``

default: ``"msranni/nni:latest"``


localStorageMountPoint
""""""""""""""""""""""

:ref:`Mount point <path>` of storage service (typically NFS) on the local machine.

type: ``str``


containerStorageMountPoint
""""""""""""""""""""""""""

Mount point of storage service (typically NFS) in docker container.

type: ``str``

This must be an absolute path.


reuseMode
"""""""""

Enable `reuse mode <../TrainingService/Overview.rst#training-service-under-reuse-mode>`__.

type: ``bool``

default: ``False``


openpaiConfig
"""""""""""""

Embedded OpenPAI config file.

type: ``Optional[JSON]``


openpaiConfigFile
"""""""""""""""""

`Path`_ to OpenPAI config file.

type: ``Optional[str]``

An example can be found `here <https://github.com/microsoft/pai/blob/master/docs/manual/cluster-user/examples/hello-world-job.yaml>`__.


AmlConfig
---------

Detailed usage can be found `here <../TrainingService/AMLMode.rst>`__.


platform
""""""""

Constant string ``"aml"``.


dockerImage
"""""""""""

Name and tag of docker image to run the trials.

type: ``str``

default: ``"msranni/nni:latest"``


subscriptionId
""""""""""""""

Azure subscription ID.

type: ``str``


resourceGroup
"""""""""""""

Azure resource group name.

type: ``str``


workspaceName
"""""""""""""

Azure workspace name.

type: ``str``


computeTarget
"""""""""""""

AML compute cluster name.

type: ``str``


HybridConfig
------------

Currently only support `LocalConfig`_, `RemoteConfig`_, :ref:`OpenpaiConfig <openpai-class>` and `AmlConfig`_ . Detailed usage can be found `here <../TrainingService/HybridMode.rst>`__.

type: list of `TrainingServiceConfig`_


SharedStorageConfig
^^^^^^^^^^^^^^^^^^^

Detailed usage can be found `here <../Tutorial/HowToUseSharedStorage.rst>`__.


nfsConfig
---------

storageType
"""""""""""

Constant string ``"NFS"``.


localMountPoint
"""""""""""""""

The path that the storage has been or will be mounted in the local machine.

type: ``str``

If the path does not exist, it will be created automatically. Recommended to use an absolute path, i.e. ``/tmp/nni-shared-storage``.


remoteMountPoint
""""""""""""""""

The path that the storage will be mounted in the remote achine.

type: ``str``

If the path does not exist, it will be created automatically. Recommended to use a relative path. i.e. ``./nni-shared-storage``.


localMounted
""""""""""""

Specify the object and status to mount the shared storage.

type: ``str``

values: ``"usermount"``, ``"nnimount"``, ``"nomount"``

``usermount`` means the user has already mounted this storage on localMountPoint. ``nnimount`` means NNI will try to mount this storage on localMountPoint. ``nomount`` means storage will not mount in the local machine, will support partial storages in the future.


nfsServer
"""""""""

NFS server host.

type: ``str``


exportedDirectory
"""""""""""""""""

Exported directory of NFS server, detailed `here <https://www.ibm.com/docs/en/aix/7.2?topic=system-nfs-exporting-mounting>`_.

type: ``str``


azureBlobConfig
---------------

storageType
"""""""""""

Constant string ``"AzureBlob"``.


localMountPoint
"""""""""""""""

The path that the storage has been or will be mounted in the local machine.

type: ``str``

If the path does not exist, it will be created automatically. Recommended to use an absolute path, i.e. ``/tmp/nni-shared-storage``.


remoteMountPoint
""""""""""""""""

The path that the storage will be mounted in the remote achine.

type: ``str``

If the path does not exist, it will be created automatically. Recommended to use a relative path. i.e. ``./nni-shared-storage``.

Note that the directory must be empty when using AzureBlob. 


localMounted
""""""""""""

Specify the object and status to mount the shared storage.

type: ``str``

values: ``"usermount"``, ``"nnimount"``, ``"nomount"``

``usermount`` means the user has already mounted this storage on localMountPoint. ``nnimount`` means NNI will try to mount this storage on localMountPoint. ``nomount`` means storage will not mount in the local machine, will support partial storages in the future.


storageAccountName
""""""""""""""""""

Azure storage account name.

type: ``str``


storageAccountKey
"""""""""""""""""

Azure storage account key.

type: ``Optional[str]``

When not set storageAccountKey, should use ``az login`` with Azure CLI at first and set `resourceGroupName`_.


resourceGroupName
"""""""""""""""""

Resource group that AzureBlob container belongs to.

type: ``Optional[str]``

Required if ``storageAccountKey`` not set.

containerName
"""""""""""""

AzureBlob container name.

type: ``str``
