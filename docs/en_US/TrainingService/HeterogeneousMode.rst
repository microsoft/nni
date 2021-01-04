**Run an Experiment on Hybrid Mode**
===========================================

Run NNI on hybrid mode means that NNI will run trials jobs in multiple kinds of training platforms. For example, NNI could submit trial jobs to remote machine and AML simultaneously。

<<<<<<< HEAD
## Setup environment
NNI has supported [local](./LocalMode.md), [remote](./RemoteMachineMode.md), [pai](./PaiMode.md) and [AML](./AMLMode.md) for hybrid training service. Before starting an experiment using these mode, users should setup the corresponding environment for the platforms. More details about the environment setup could be found in the corresponding docs.
=======
Setup environment
-----------------
>>>>>>> 5d177d694d6ab96fc07bd0bcd9fdba1e7a440486

NNI has supported `local <./LocalMode.rst>`__\ , `remote <./RemoteMachineMode.rst>`__\ , `PAI <./PaiMode.rst>`__\ , and `AML <./AMLMode.rst>`__ for heterogeneous training service. Before starting an experiment using these mode, users should setup the corresponding environment for the platforms. More details about the environment setup could be found in the corresponding docs.

Run an experiment
-----------------

Use ``examples/trials/mnist-tfv1`` as an example. The NNI config YAML file's content is like:

.. code-block:: yaml

    authorName: default
    experimentName: example_mnist
    trialConcurrency: 2
    maxExecDuration: 1h
    maxTrialNum: 10
    trainingServicePlatform: hybrid
    searchSpacePath: search_space.json
    #choice: true, false
    useAnnotation: false
    tuner:
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 mnist.py
      codeDir: .
      gpuNum: 1
    hybridConfig:
      trainingServicePlatforms:
        - local
        - remote
    remoteConfig:
      reuse: true
    machineList:
      - ip: 10.1.1.1
        username: bob
        passwd: bob123



Configurations for hybrid mode:
hybridConfig:
* trainingServicePlatforms. required key. This field specify the platforms used in hybrid mode, the values using yaml list format. NNI support setting `local`, `remote`, `aml`, `pai` in this field.

* trainingServicePlatforms. required key. This field specify the platforms used in heterogeneous mode, the values using yaml list format. NNI support setting ``local``, ``remote``, ``aml``, ``pai`` in this field.


.. Note:: If setting a platform in trainingServicePlatforms mode, users should also set the corresponding configuration for the platform. For example, if set ``remote`` as one of the platform, should also set ``machineList`` and ``remoteConfig`` configuration.
