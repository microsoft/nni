**Run an Experiment on Hybrid Mode**
===========================================

Run NNI on hybrid mode means that NNI will run trials jobs in multiple kinds of training platforms. For example, NNI could submit trial jobs to remote machine and AML simultaneously.

Setup environment
-----------------

NNI has supported `local <./LocalMode.rst>`__\ , `remote <./RemoteMachineMode.rst>`__\ , `PAI <./PaiMode.rst>`__\ , `AML <./AMLMode.rst>`__,  `Kubeflow <./KubeflowMode.rst>`__\ , `FrameworkController <./FrameworkControllerMode.rst>`__\ ,for hybrid training service. Before starting an experiment using these mode, users should setup the corresponding environment for the platforms. More details about the environment setup could be found in the corresponding docs.

Run an experiment
-----------------

Use ``examples/trials/mnist-tfv1`` as an example. The NNI config YAML file's content is like:

.. code-block:: yaml

    experimentName: MNIST
    searchSpaceFile: search_space.json
    trialCommand: python3 mnist.py
    trialCodeDirectory: .
    trialConcurrency: 2
    trialGpuNumber: 0
    maxExperimentDuration: 24h
    maxTrialNumber: 100
    tuner:
      name: TPE
      classArgs:
        optimize_mode: maximize
    trainingService:
      - platform: remote
        machineList:
          - host: 127.0.0.1
            user: bob
            password: bob
      - platform: local

To use hybrid training services, users should set training service configurations as a list in `trainingService` field.  
Currently, hybrid support setting `local`, `remote`, `pai`, `aml`, `kubeflow` and `frameworkcontroller` training services.
