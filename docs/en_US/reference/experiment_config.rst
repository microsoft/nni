===========================
Experiment Config Reference
===========================

This is the detailed list of experiment config fields.
For quick start guide, use tutorial instead.

Notes
=====

1. This document list field names as separated words.
   They should be spelled in ``snake_case`` for Python library ``nni.experiment``, and are normally spelled in ``camelCase`` for YAML files.

2. In this document type of fields are formatted as `Python type hint <https://docs.python.org/3.10/library/typing.html>`__.
   Therefore JSON objects are called `dict` and arrays are called `list`.

.. _path:

3. Some fields take a path to file or directory.
   Unless otherwise noted, both absolute path and relative path are supported, and ``~`` will be expanded to home directory.

   - When written in YAML file, relative paths are relative to the directory containing that file.
   - When assigned in Python code, relative paths are relative to current working directory.
   - All relative paths are converted to absolute when loading YAML file into Python class, and when saving Python class to YAML file.

4. Setting a field to ``None`` or ``null`` is equivalent to not setting the field.

ExperimentConfig
================

experiment name
---------------

Mnemonic name of the experiment. This will be shown in web UI and nnictl.

type: ``Optional[str]``


search space file
-----------------

Path_ to a JSON file containing the search space.

type: ``Optional[str]``

Search space format is determined by tuner. Common format for built-in tuners is documeted `here <../Tutorial/SearchSpaceSpec.rst>`__.

Mutually exclusive to `search space`_.


search space
------------

Search space object.

type: ``Optional[JSON]``

The format is determined by tuner. Common format for built-in tuners is documented `here <../Tutorial/SearchSpaceSpec.rst>`__.

Note that ``None`` means "no such field" so empty search space should be written as ``{}``.

Mutually exclusive to `search space file`_.


trial command
-------------

Command to launch trial.

type: ``str``

The command will be executed in bash on Linux and macOS, and in PowerShell on Windows.


trial code directory
--------------------

`Path`_ to the directory containing trial source files.

type: ``str``

default: ``"."``

All files in this directory will be sent to training machine, unless there is a `.nniignore file <../Tutorial/QuickStart.rst#nniignore>`__.


trial concurrency
-----------------

Specify how many trials should be run concurrently.

type: ``int``

The real concurrency also depends on hardware resources and may be less than this value.


trial gpu number
----------------

Number of GPUs used by each trial.

type: ``Optional[int]``

This field might have slightly different meaning for various training services,
especially when set to ``0`` or ``None``.
See training service's document for details.

In local mode, setting the field to zero will prevent trials from accessing GPU (by empty ``CUDA_VISIBLE_DEVICES``).
And when set to ``None``, trials will be created and scheduled as if they did not use GPU,
but they can still use all GPU resources if they want.


max experiment duration
-----------------------

Limit the duration of this experiment if specified.

type: ``Optional[str]``

format: ``number + s|m|h|d``

examples: ``"10m"``, ``"0.5h"``

When time runs out, the experiment will stop creating trials but continue to serve web UI.


max trial number
----------------

Limit the number of trials to create if specified.

type: ``Optional[int]``

When the budget runs out, the experiment will stop creating trials but continue to serve web UI.


nni manager ip
--------------

IP of current machine, used by training machines to access NNI manager. Not used in local mode.

type: ``Optional[str]``

If not specified, IPv4 address of ``eth0`` will be used.

Must be set on Windows and systems using predictable network interface name, except for local mode.


use annotation
--------------

Enable `annotation <../Tutorial/AnnotationSpec.rst>`__.

type: ``bool``

default: ``False``

When using annotation, `search space`_ and `search space file`_ should not be specified manually.


debug
-----

Enable debug mode.

type: ``bool``

default: ``False``

When enabled, logging will be more verbose and some internal validation will be loosen.


log level
---------

Set log level of whole system.

type: ``Optional[str]``

values: ``"trace"``, ``"debug"``, ``"info"``, ``"warning"``, ``"error"``, ``"fatal"``

Defaults to "info" or "debug", depending on `debug`_ option.

Most modules of NNI will be affected by this value, including NNI manager, tuner, training service, etc.

The exception is trial, whose logging level is directly managed by trial code.

For Python modules, "trace" acts as logging level 0 and "fatal" acts as ``logging.CRITICAL``.


experiment working directory
----------------------------

Specify the `directory <path>`_ to place log, checkpoint, metadata, and other run-time stuff.

type: ``Optional[str]``

By default uses ``~/nni-experiments``.

NNI will create a subdirectory named by experiment ID, so it is safe to use same directory for multiple experiments.


tuner gpu indices
-----------------

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


training service
----------------

Specify `training service <../TrainingService/Overview.rst>`__.

type: `TrainingServiceConfig`_


AlgorithmConfig
===============

``AlgorithmConfig`` describes a tuner / assessor / advisor algorithm.

For custom algorithms, there are two ways to describe them:

  1. `Register the algorithm <../Tuner/InstallCustomizedTuner.rst>`__ to use it like built-in. (preferred)

  2. Specify code directory and class name directly.


name
----

Name of built-in or registered algorithm.

type: ``str`` for built-in and registered algorithm, ``None`` for other custom algorithm


class name
----------

Qualified class name of not registered custom algorithm.

type: ``None`` for built-in and registered algorithm, ``str`` for other custom algorithm

example: ``"my_tuner.MyTuner"``


code directory
--------------

`Path`_ to directory containing the custom algorithm class.

type: ``None`` for built-in and registered algorithm, ``str`` for other custom algorithm


class args
----------

Keyword arguments passed to algorithm class' constructor.

type: ``Optional[dict[str, Any]]``

See algorithm's document for supported value.


TrainingServiceConfig
=====================

One of following:

- `LocalConfig`_
- `RemoteConfig`_
- `OpenpaiConfig`_
- `AmlConfig`_

For other training services, we suggest to use `v1 config schema <../Tutorial/ExperimentConfig.rst>`_ for now.


LocalConfig
===========

Detailed `here <../TrainingService/LocalMode.rst>`__.

platform
--------

Constant string ``"local"``.


use active gpu
--------------

Specify whether NNI should submit trials to GPUs occupied by other tasks.

type: ``Optional[bool]``

Must be set when `trial gpu number` greater than zero.

If your are using desktop system with GUI, set this to ``True``.


max trial number per gpu
------------------------

Specify how many trials can share one GPU.

type: ``int``

default: ``1``


gpu indices
-----------

Limit the GPUs visible to trial processes.

type: ``Optional[list[int] | str]``

If `trial gpu number`_ is less than the length of this value, only a subset will be visible to each trial.

This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.


RemoteConfig
============

Detailed `here <../TrainingService/RemoteMachineMode.rst>`__.

platform
--------

Constant string ``"remote"``.


machine list
------------

List of training machines.

type: list of `RemoteMachineConfig`_


reuse mode
----------

Enable reuse `mode <../Tutorial/ExperimentConfig.rst#reuse>`__.

type: ``bool``


RemoteMachineConfig
===================

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

If not specified, `ssh key file`_ will be used instead.


ssh key file
------------

`Path`_ to ssh key file (identity file).

type: ``Optional[str]``

Only used when `password`_ is not specified.


ssh passphrase
--------------

Passphrase of SSH identity file.

type: ``Optional[str]``


use active gpu
--------------

Specify whether NNI should submit trials to GPUs occupied by other tasks.

type: ``bool``

default: ``False``


max trial number per gpu
------------------------

Specify how many trials can share one GPU.

type: ``int``

default: ``1``


gpu indices
-----------

Limit the GPUs visible to trial processes.

type: ``Optional[list[int] | str]``

If `trial gpu number`_ is less than the length of this value, only a subset will be visible to each trial.

This will be used as ``CUDA_VISIBLE_DEVICES`` environment variable.


trial prepare command
---------------------

Command(s) to run before launching each trial.

type: ``Optional[str]``

This is useful if preparing steps vary for different machines.


OpenpaiConfig
=============

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


docker image
------------

Name and tag of docker image to run the trials.

type: ``str``

default: ``"msranni/nni:latest"``


nni manager storage mount point
-------------------------------

`Mount point <path>`_ of storage service (typically NFS) on current machine.

type: ``str``


container storage mount point
-----------------------------

Mount point of storage service (typically NFS) in docker container.

type: ``str``

This must be an absolute path.


reuse mode
----------

Enable reuse `mode <../Tutorial/ExperimentConfig.rst#reuse>`__.

type: ``bool``

default: ``False``


open pai config
---------------

Embedded OpenPAI config file.

type: ``Optional[JSON]``


open pai config file
--------------------

`Path`_ to OpenPAI config file.

type: ``Optional[str]``

An example can be found `here <https://github.com/microsoft/pai/blob/master/docs/manual/cluster-user/examples/hello-world-job.yaml>`__


AmlConfig
=========

Detailed `here <../TrainingService/AMLMode.rst>`__.


platform
--------

Constant string ``"aml"``.


docker image
------------

Name and tag of docker image to run the trials.

type: ``str``

default: ``"msranni/nni:latest"``


subscription id
---------------

Azure subscription ID.

type: ``str``


resource group
--------------

Azure resource group name.

type: ``str``


workspace name
--------------

Azure workspace name.

type: ``str``


compute target
--------------

AML compute cluster name.

type: ``str``
