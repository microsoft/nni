**以混合模式进行实验**
===========================================

在混合模式下运行 NNI 意味着 NNI 将在多种培训平台上运行试验工作。 例如，NNI 可以同时将试用作业提交到远程计算机和 AML。

设置环境
-----------------

对于混合模式，NNI 目前支持的平台有 `本地平台 <LocalMode.rst>`__\ ，`远程平台 <RemoteMachineMode.rst>`__\ ， `PAI <PaiMode.rst>`__ 和 `AML <./AMLMode.rst>`__\ 。 使用这些模式开始 Experiment 之前，用户应为平台设置相应的环境。 有关环境设置的详细信息，请参见相应的文档。

运行实验
-----------------

以 ``examples/trials/mnist-tfv1`` 为例。 NNI 的 YAML 配置文件如下：

.. code-block:: yaml

    authorName: default
    experimentName: example_mnist
    trialConcurrency: 2
    maxExecDuration: 1h
    maxTrialNum: 10
    trainingServicePlatform: hybrid
    searchSpacePath: search_space.json
    # 可选项：true, false
    useAnnotation: false
    tuner:
      builtinTunerName: TPE
      classArgs:
        # 可选项: maximize, minimize
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

混合模式的配置：

hybridConfig:

* trainingServicePlatforms. 必填。 该字段指定用于混合模式的平台，值使用 yaml 列表格式。 NNI 支持在此字段中设置 ``local``, ``remote``, ``aml``, ``pai`` 。


.. Note:: 如果将平台设置为 trainingServicePlatforms 模式，则用户还应该为平台设置相应的配置。 例如，如果使用 ``remote`` 作为平台，还应设置 ``machineList`` 和 ``remoteConfig`` 配置。