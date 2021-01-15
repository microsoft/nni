**在异构模式下运行 Experiment**
===========================================

在异构模式下运行 NNI 意味着 NNI 将同时在多种培训平台上运行试验工作。 例如，NNI 可以同时将试用作业提交到远程计算机和 AML。

设置环境
----------------------

NNI 的异构模式目前支持 `local <./LocalMode.rst>`__\ , `remote <./RemoteMachineMode.rst>`__\ , `PAI <./PaiMode.rst>`__\ 和 `AML <./AMLMode.rst>`__ 四种训练环境。在使用这些模式开始实验之前，应在平台上设置对应的环境。环境设置的详细信息，参见以上文档。


运行实验
--------------------

以 `examples/trials/mnist-tfv1` 为例。 NNI 的 YAML 配置文件如下：

.. code-block:: yaml

    authorName: default
    experimentName: example_mnist
    trialConcurrency: 2
    maxExecDuration: 1h
    maxTrialNum: 10
    trainingServicePlatform: heterogeneous
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
    heterogeneousConfig:
      trainingServicePlatforms:
        - local
        - remote
    remoteConfig:
      reuse: true
    machineList:
      - ip: 10.1.1.1
        username: bob
        passwd: bob123

异构模式的配置：

heterogeneousConfig:

* trainingServicePlatforms. 必填。 该字段指定用于异构模式的平台，值使用 yaml 列表格式。 NNI 支持在此字段中设置 `local`, `remote`, `aml`, `pai` 。


.. Note:: 如果将平台设置为 trainingServicePlatforms 模式，则用户还应该为平台设置相应的配置。 例如，如果使用 ``remote`` 作为平台，还应设置 ``machineList`` 和 ``remoteConfig`` 配置。
