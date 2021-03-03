===========================
Experiment Config Reference
===========================

Notes
=====

1. This document list field names is ``camelCase``.
   They need to be converted to ``snake_case`` for Python library ``nni.experiment``.

2. In this document type of fields are formatted as `Python type hint <https://docs.python.org/3.10/library/typing.html>`__.
   Therefore JSON objects are called `dict` and arrays are called `list`.

.. _path:

3. Some fields take a path to file or directory.
   Unless otherwise noted, both absolute path and relative path are supported, and ``~`` will be expanded to home directory.

   - When written in YAML file, relative paths are relative to the directory containing that file.
   - When assigned in Python code, relative paths are relative to current working directory.
   - All relative paths are converted to absolute when loading YAML file into Python class, and when saving Python class to YAML file.

4. Setting a field to ``None`` or ``null`` is equivalent to not setting the field.

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

Mnemonic name of the experiment. This will be shown in web UI and nnictl.

type: ``Optional[str]``


searchSpaceFile
---------------

Path_ to a JSON file containing the search space.

type: ``Optional[str]``

Search space format is determined by tuner. Common format for built-in tuners is documeted `here <../Tutorial/SearchSpaceSpec.rst>`__.

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


trialCodeDirectory
------------------

`Path`_ to the directory containing trial source files.

type: ``str``

default: ``"."``

All files in this directory will be sent to training machine, unless there is a ``.nniignore`` file.
(See nniignore section of `quick start guide <../Tutorial/QuickStart.rst>`__ for details.)


trialConcurrency
----------------

Specify how many trials should be run concurrently.

type: ``int``

The real concurrency also depends on hardware resources and may be less than this value.


trialGpuNumber
--------------

Number of GPUs used by each trial.

type: ``Optional[int]``

This field might have slightly different meaning for various training services,
especially when set to ``0`` or ``None``.
See training service's document for details.

In local mode, setting the field to zero will prevent trials from accessing GPU (by empty ``CUDA_VISIBLE_DEVICES``).
And when set to ``None``, trials will be created and scheduled as if they did not use GPU,
but they can still use all GPU resources if they want.


maxExperimentDuration
---------------------

Limit the duration of this experiment if specified.

type: ``Optional[str]``

format: ``number + s|m|h|d``

examples: ``"10m"``, ``"0.5h"``

When time runs out, the experiment will stop creating trials but continue to serve web UI.


maxTrialNumber
--------------

Limit the number of trials to create if specified.

type: ``Optional[int]``

When the budget runs out, the experiment will stop creating trials but continue to serve web UI.


nniManagerIp
------------

IP of current machine, used by training machines to access NNI manager. Not used in local mode.

type: ``Optional[str]``

If not specified, IPv4 address of ``eth0`` will be used.

Must be set on Windows and systems using predictable network interface name, except for local mode.


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

When enabled, logging will be more verbose and some internal validation will be loosen.


logLevel
--------

Set log level of whole system.

type: ``Optional[str]``

values: ``"trace"``, ``"debug"``, ``"info"``, ``"warning"``, ``"error"``, ``"fatal"``

Defaults to "info" or "debug", depending on `debug`_ option.

Most modules of NNI will be affected by this value, including NNI manager, tuner, training service, etc.

The exception is trial, whose logging level is directly managed by trial code.

For Python modules, "trace" acts as logging level 0 and "fatal" acts as ``logging.CRITICAL``.


experimentWorkingDirectory
--------------------------

Specify the `directory <path>`_ to place log, checkpoint, metadata, and other run-time stuff.

type: ``Optional[str]``

By default uses ``~/nni-experiments``.

NNI will create a subdirectory named by experiment ID, so it is safe to use same directory for multiple experiments.


tunerGpuIndices
---------------

Limit the GPUs visible to tuner, assessor, and advisor.

type: ``Optional[list[int] | str]``

This will be the ``CUDA_VISIBLE_DEVICES`` environment variable of tuner process.

Because tuner, assessor, and advisor run in same process, this option will affect them all.


tuner
-----

Specify the tuner.

type: Optional `AlgorithmConfig`_


assessor
--------

Specify the assessor.

type: Optional `AlgorithmConfig`_


advisor
-------

Specify the advisor.

type: Optional `AlgorithmConfig`_


trainingService
---------------

Specify `training service <../TrainingService/Overview.rst>`__.

type: `TrainingServiceConfig`_


AlgorithmConfig
^^^^^^^^^^^^^^^

``AlgorithmConfig`` describes a tuner / assessor / advisor algorithm.

For custom algorithms, there are two ways to describe them:

  1. `Register the algorithm <../Tuner/InstallCustomizedTuner.rst>`__ to use it like built-in. (preferred)

  2. Specify code directory and class name directly.


name
----

Name of built-in or registered algorithm.

type: ``str`` for built-in and registered algorithm, ``None`` for other custom algorithm


className
---------

Qualified class name of not registered custom algorithm.

type: ``None`` for built-in and registered algorithm, ``str`` for other custom algorithm

example: ``"my_tuner.MyTuner"``


codeDirectory
-------------

`Path`_ to directory containing the custom algorithm class.

type: ``None`` for built-in and registered algorithm, ``str`` for other custom algorithm


classArgs
---------

Keyword arguments passed to algorithm class' constructor.

type: ``Optional[dict[str, Any]]``

See algorithm's document for supported value.


TrainingServiceConfig
^^^^^^^^^^^^^^^^^^^^^

One of following:

- `LocalConfig`_
- `RemoteConfig`_
- `OpenpaiConfig <openpai-class>`_
- `AmlConfig`_

For other training services, we suggest to use `v1 config schema <../Tutorial/ExperimentConfig.rst>`_ for now.


LocalConfig
^^^^^^^^^^^

Detailed `here <../TrainingService/LocalMode.rst>`__.

platform
--------

Constant string ``"local"``.


useActiveGpu
------------

Specify whether NNI should submit trials to GPUs occupied by other tasks.

type: ``Optional[bool]``

Must be set when `trialGpuNumber` greater than zero.

If your are using desktop system with GUI, set this to ``True``.


maxTrialNumberPerGpu
---------------------

Specify how many trials can share one GPU.

type: ``int``

default: ``1``


gpuIndices
----------

Limit the GPUs visible to trial processes.

type: ``Optional[list[int] | str]``

If `trialGpuNumber`_ is less than the length of this value, only a subset will be visible to each trial.

This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.


RemoteConfig
^^^^^^^^^^^^

Detailed `here <../TrainingService/RemoteMachineMode.rst>`__.

platform
--------

Constant string ``"remote"``.


machineList
-----------

List of training machines.

type: list of `RemoteMachineConfig`_


reuseMode
---------

Enable reuse `mode <../Tutorial/ExperimentConfig.rst#reuse>`__.

type: ``bool``


RemoteMachineConfig
^^^^^^^^^^^^^^^^^^^

host
----

IP or hostname (domain name) of the machine.

type: ``str``


port
----

SSH service port.

type: ``int``

default: ``22``


user
----

Login user name.

type: ``str``


password
--------

Login password.

type: ``Optional[str]``

If not specified, `sshKeyFile`_ will be used instead.


sshKeyFile
----------

`Path`_ to sshKeyFile (identity file).

type: ``Optional[str]``

Only used when `password`_ is not specified.


sshPassphrase
-------------

Passphrase of SSH identity file.

type: ``Optional[str]``


useActiveGpu
------------

Specify whether NNI should submit trials to GPUs occupied by other tasks.

type: ``bool``

default: ``False``


maxTrialNumberPerGpu
--------------------

Specify how many trials can share one GPU.

type: ``int``

default: ``1``


gpuIndices
----------

Limit the GPUs visible to trial processes.

type: ``Optional[list[int] | str]``

If `trialGpuNumber`_ is less than the length of this value, only a subset will be visible to each trial.

This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.


trialPythonPath
-------------------

Command(s) to run before launching each trial.

type: ``Optional[str]``

This is useful if preparing steps vary for different machines.

.. _openpai-class:

OpenpaiConfig
^^^^^^^^^^^^^

Detailed `here <../TrainingService/PaiMode.rst>`__.

platform
--------

Constant string ``"openpai"``.


host
----

Hostname of OpenPAI service.

type: ``str``

This may includes ``https://`` or ``http://`` prefix.

HTTPS will be used by default.


username
--------

OpenPAI user name.

type: ``str``


token
-----

OpenPAI user token.

type: ``str``

This can be found in your OpenPAI user settings page.


dockerImage
-----------

Name and tag of docker image to run the trials.

type: ``str``

default: ``"msranni/nni:latest"``


nniManagerStorageMountPoint
---------------------------

`Mount point <path>`_ of storage service (typically NFS) on current machine.

type: ``str``


containerStorageMountPoint
--------------------------

Mount point of storage service (typically NFS) in docker container.

type: ``str``

This must be an absolute path.


reuseMode
---------

Enable reuse `mode <../Tutorial/ExperimentConfig.rst#reuse>`__.

type: ``bool``

default: ``False``


openpaiConfig
-------------

Embedded OpenPAI config file.

type: ``Optional[JSON]``


openpaiConfigFile
-----------------

`Path`_ to OpenPAI config file.

type: ``Optional[str]``

An example can be found `here <https://github.com/microsoft/pai/blob/master/docs/manual/cluster-user/examples/hello-world-job.yaml>`__


AmlConfig
^^^^^^^^^

Detailed `here <../TrainingService/AMLMode.rst>`__.


platform
--------

Constant string ``"aml"``.


dockerImage
-----------

Name and tag of docker image to run the trials.

type: ``str``

default: ``"msranni/nni:latest"``


subscriptionId
--------------

Azure subscription ID.

type: ``str``


resourceGroup
-------------

Azure resource group name.

type: ``str``


workspaceName
-------------

Azure workspace name.

type: ``str``


computeTarget
-------------

AML compute cluster name.

type: ``str``
