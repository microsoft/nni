Local Training Service
======================

With local training service, the whole experiment (e.g., tuning algorithms, trials) runs on a single machine, i.e., user's dev machine. The generated trials run on this machine following ``trialConcurrency`` set in the configuration yaml file. If GPUs are used by trial, local training service will allocate required number of GPUs for each trial, like a resource scheduler.

.. note:: Currently, :ref:`reuse mode <training-service-reuse>` remains disabled by default in local training service.

Prerequisite
------------

You are recommended to go through quick start first, as this document page only explains the configuration of local training service, one part of the experiment configuration yaml file.


Usage
-----

..  code-block:: yaml

    # the experiment config yaml file
    ...
    trainingService:
      platform: local
      useActiveGpu: false # optional
    ...

There are other supported fields for local training service, such as ``maxTrialNumberPerGpu``, ``gpuIndices``, for concurrently running multiple trials on one GPU, and running trials on a subset of GPUs on your machine. Please refer to :ref:`reference-local-config-label` in reference for detailed usage.

..  note::
    Users should set **useActiveGpu** to `true`, if the local machine has GPUs and your trial uses GPU, but generated trials keep waiting. This is usually the case when you are using graphical OS like Windows 10 and Ubuntu desktop.

Then we explain how local training service works with different configurations of ``trialGpuNumber`` and ``trialConcurrency``. Suppose user's local machine has 4 GPUs, with configuration ``trialGpuNumber: 1`` and ``trialConcurrency: 4``, there will be 4 trials run on this machine concurrently, each of which uses 1 GPU. If the configuration is ``trialGpuNumber: 2`` and ``trialConcurrency: 2``, there will be 2 trials run on this machine concurrently, each of which uses 2 GPUs. Which GPU is allocated to which trial is decided by local training service, users do not need to worry about it. An exmaple configuration below.

..  code-block:: yaml

    ...
    trialGpuNumber: 1
    trialConcurrency: 4
    ...
    trainingService:
      platform: local
      useActiveGpu: false

A complete example configuration file can be found :githublink:`examples/trials/mnist-pytorch/config.yml`.