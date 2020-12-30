**以异构模式进行实验**
===========================================

在异构模式下运行 NNI 意味着 NNI 将在多种培训平台上运行试验工作。 例如，NNI 可以同时将试用作业提交到远程计算机和 AML。

## 设置环境
NNI has supported [local](./LocalMode.md), [remote](./RemoteMachineMode.md), [pai](./PaiMode.md) and [AML](./AMLMode.md) for heterogeneous training service. Before starting an experiment using these mode, users should setup the corresponding environment for the platforms. More details about the environment setup could be found in the corresponding docs.



## 运行实验
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


注意：  
    如果将平台设置为 trainingServicePlatforms 模式，则用户还应该为平台设置相应的配置。 例如，如果使用 ``remote`` 作为平台，还应设置 ``machineList`` 和 ``remoteConfig`` 配置。
