===========================
Experiment（实验）配置参考
===========================

注意
=====

1. 此文档的字段使用 ``camelCase`` 法命名。
   对于 Python 库 ``nni.experiment``，需要转换成 ``snake_case`` 形式。

2. 在此文档中，字段类型被格式化为 `Python 类型提示 <https://docs.python.org/3.10/library/typing.html>`__。
   因此，JSON 对象被称为 `dict`，数组被称为 `list`。

.. _路径:

3. 一些字段采用文件或目录的路径，
   除特别说明，均支持绝对路径和相对路径，``~`` 将扩展到 home 目录。

   - 在写入 YAML 文件时，相对路径是相对于包含该文件目录的路径。
   - 在 Python 代码中赋值时，相对路径是相对于当前工作目录的路径。
   - 在将 YAML 文件加载到 Python 类，以及将 Python 类保存到 YAML 文件时，所有相对路径都转换为绝对路径。

4. 将字段设置为 ``None`` 或 ``null`` 时相当于不设置该字段。

示例
========

本机模式
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

本机模式（内联搜索空间）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

远程模式
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

参考
=========

Experiment 配置
^^^^^^^^^^^^^^^^

experimentName
--------------

Experiment 的助记名称， 这将显示在 WebUI 和 nnictl 中。

类型：``Optional[str]``


searchSpaceFile
---------------

包含搜索空间 JSON 文件的\ 路径_ 。

类型：``Optional[str]``

搜索空间格式由 Tuner 决定， 内置 Tuner 的通用格式在 `这里 <../Tutorial/SearchSpaceSpec.rst>`__。

与 `searchSpace`_ 互斥。


searchSpace
-----------

搜索空间对象。

类型：``Optional[JSON]``

格式由 Tuner 决定， 内置 Tuner 的通用格式在 `这里 <../Tutorial/SearchSpaceSpec.rst>`__。

注意，``None`` 意味着“没有这样的字段”，所以空的搜索空间应该写成 ``{}``。

与 `searchSpaceFile`_ 互斥。


trialCommand
------------

启动 Trial 的命令。

类型：``str``

该命令将在 Linux 和 macOS 上的 bash 中执行，在 Windows 上的 PowerShell 中执行。


trialCodeDirectory
------------------

到 Trial 源文件的目录的 路径_。

类型：``str``

默认值：``"."``

此目录中的所有文件都将发送到训练机器，除了 ``.nniignore`` 文件。
（详细信息，请参考 `快速入门 <../Tutorial/QuickStart.rst>`__ 的 nniignore 部分。）


trialConcurrency
----------------

指定同时运行的 Trial 数目。

类型：``int``

实际的并发性还取决于硬件资源，可能小于此值。


trialGpuNumber
--------------

每个 Trial 使用的 GPU 数目。

类型：``Optional[int]``

对于各种训练平台，这个字段的含义可能略有不同，
尤其是设置为 ``0`` 或者 ``None`` 时，
详情请参阅训练平台文件。

在本地模式下，将该字段设置为零将阻止 Trial 获取 GPU（通过置空 ``CUDA_VISIBLE_DEVICES`` ）。
当设置为 ``None`` 时，Trial 将被创建和调度，就像它们不使用 GPU 一样，
但是它们仍然可以根据需要使用所有 GPU 资源。


maxExperimentDuration
---------------------

如果指定，将限制此 Experiment 的持续时间。

类型：``Optional[str]``

格式：``数字 + s|m|h|d``

示例：``"10m"``, ``"0.5h"``

当时间耗尽时，Experiment 将停止创建 Trial，但仍然服务于 web UI。


maxTrialNumber
--------------

如果指定，将限制创建的 Trial 数目。

类型：``Optional[int]``

当预算耗尽时，Experiment 将停止创建 Trial，但仍然服务于 web UI。


nniManagerIp
------------

当前机器的 IP，用于训练机器访问 NNI 管理器。 本机模式下不可选。

类型：``Optional[str]``

如果未指定，将使用 ``eth0`` 的 IPv4 地址。

必须在 Windows 和使用可预测网络接口名称的系统上设置，本地模式除外。


useAnnotation
-------------

启动 `annotation <../Tutorial/AnnotationSpec.rst>`__。

类型：``bool``

默认值：``false``

使用 annotation 时，`searchSpace`_ 和 `searchSpaceFile`_ 不应手动指定。


debug
-----

启动调试模式

类型：``bool``

默认值：``false``

启用后，日志记录将更加详细，并且一些内部验证将被放宽。


logLevel
--------

设置整个系统的日志级别。

类型：``Optional[str]``

候选项：``"trace"``, ``"debug"``, ``"info"``, ``"warning"``, ``"error"``, ``"fatal"``

默认为 "info" 或 "debug"，取决于 `debug`_ 选项。

NNI 的大多数模块都会受到此值的影响，包括 NNI 管理器、Tuner、训练平台等。

Trial 是一个例外，它的日志记录级别由 Trial 代码直接管理。

对于 Python 模块，"trace" 充当日志级别0，"fatal" 表示 ``logging.CRITICAL``。


experimentWorkingDirectory
--------------------------

指定目录 `directory <path>`_ 来存放日志、检查点、元数据和其他运行时的内容。

类型：``Optional[str]``

默认：``~/nni-experiments``

NNI 将创建一个以 Experiment ID 命名的子目录，所以在多个 Experiment 中使用同一个目录不会有冲突。


tunerGpuIndices
---------------

设定对 Tuner、Assessor 和 Advisor 可见的 GPU。

类型：``Optional[list[int] | str]``

这将是 Tuner 进程的 ``CUDA_VISIBLE_DEVICES`` 环境变量，

因为 Tuner、Assessor 和 Advisor 在同一个进程中运行，所以此选项将同时影响它们。


tuner
-----

指定 Tuner。

类型：Optional `AlgorithmConfig`_


assessor
--------

指定 Assessor。

类型：Optional `AlgorithmConfig`_


advisor
-------

指定 Advisor。

类型：Optional `AlgorithmConfig`_


trainingService
---------------

指定 `训练平台 <../TrainingService/Overview.rst>`__。

类型：`TrainingServiceConfig`_


AlgorithmConfig
^^^^^^^^^^^^^^^

``AlgorithmConfig`` 描述 tuner / assessor / advisor 算法。

对于自定义算法，有两种方法来描述它们：

  1. `注册算法 <../Tuner/InstallCustomizedTuner.rst>`__ ，像内置算法一样使用。 （首选）

  2. 指定代码目录和类名。


name
----

内置或注册算法的名称。

类型：对于内置和注册算法使用 ``str``，其他自定义算法使用 ``None``


className
---------

未注册的自定义算法的限定类名。

类型：对于内置和注册算法使用 ``None``，其他自定义算法使用 ``str``

示例：``"my_tuner.MyTuner"``


codeDirectory
-------------

到自定义算法类的目录的 路径_。

类型：对于内置和注册算法使用 ``None``，其他自定义算法使用 ``str``


classArgs
---------

传递给算法类构造函数的关键字参数。

类型：``Optional[dict[str, Any]]``

有关支持的值，请参阅算法文档。


TrainingServiceConfig
^^^^^^^^^^^^^^^^^^^^^

以下之一：

- `LocalConfig`_
- `RemoteConfig`_
- `OpenpaiConfig <openpai-class>`_
- `AmlConfig`_

对于其他训练平台，目前 NNI 建议使用 `v1 配置模式 <../Tutorial/ExperimentConfig.rst>`_ 。


LocalConfig
^^^^^^^^^^^

详情查看 `这里 <../TrainingService/LocalMode.rst>`__。

platform
--------

字符串常量 ``"local"``。


useActiveGpu
------------

指定 NNI 是否应向被其他任务占用的 GPU 提交 Trial。

类型：``Optional[bool]``

必须在 ``trialgpunmber`` 大于零时设置。

如果您使用带有 GUI 的桌面系统，请将其设置为 ``True``。


maxTrialNumberPerGpu
---------------------

指定可以共享一个 GPU 的 Trial 数目。

类型：``int``

默认值：``1``


gpuIndices
----------

设定对 Trial 进程可见的 GPU。

类型：``Optional[list[int] | str]``

如果 `trialGpuNumber`_ 小于此值的长度，那么每个 Trial 只能看到一个子集。

这用作环境变量 ``CUDA_VISIBLE_DEVICES``。


RemoteConfig
^^^^^^^^^^^^

详情查看 `这里 <../TrainingService/RemoteMachineMode.rst>`__。

platform
--------

字符串常量 ``"remote"``。


machineList
-----------

训练机器列表

类型： `RemoteMachineConfig`_ 列表


reuseMode
---------

启动 `重用模式 <../Tutorial/ExperimentConfig.rst#reuse>`__。

类型：``bool``


RemoteMachineConfig
^^^^^^^^^^^^^^^^^^^

host
----

机器的 IP 或主机名（域名）。

类型：``str``


port
----

SSH 服务端口。

类型：``int``

默认值：``22``


user
----

登录用户名。

类型：``str``


password
--------

登录密码。

类型：``Optional[str]``

如果未指定，则将使用 `sshKeyFile`_。


sshKeyFile
----------

到 sshKeyFile的 路径_ 。

类型：``Optional[str]``

仅在未指定 `password`_ 时使用。


sshPassphrase
-------------

SSH 标识文件的密码。

类型：``Optional[str]``


useActiveGpu
------------

指定 NNI 是否应向被其他任务占用的 GPU 提交 Trial。

类型：``bool``

默认值：``false``


maxTrialNumberPerGpu
--------------------

指定可以共享一个 GPU 的 Trial 数目。

类型：``int``

默认值：``1``


gpuIndices
----------

设定对 Trial 进程可见的 GPU。

类型：``Optional[list[int] | str]``

如果 `trialGpuNumber`_ 小于此值的长度，那么每个 Trial 只能看到一个子集。

这用作环境变量 ``CUDA_VISIBLE_DEVICES``。


trialPrepareCommand
-------------------

启动 Trial 之前运行的命令。

类型：``Optional[str]``

如果不同机器的准备步骤不同，这将非常有用。

.. _openpai-class:

OpenpaiConfig
^^^^^^^^^^^^^

详情查看 `这里 <../TrainingService/PaiMode.rst>`__。

platform
--------

字符串常量 ``"openpai"``。


host
----

OpenPAI 平台的主机名。

类型：``str``

可能包括 ``https://`` 或 ``http://`` 前缀。

默认情况下将使用 HTTPS。


username
--------

OpenPAI 用户名。

类型：``str``


token
-----

OpenPAI 用户令牌。

类型：``str``

这可以在 OpenPAI 用户设置页面中找到。


dockerImage
-----------

运行 Trial 的 Docker 镜像的名称和标签。

类型：``str``

默认：``"msranni/nni:latest"``


nniManagerStorageMountPoint
---------------------------

当前机器中存储服务（通常是NFS）的挂载点路径。

类型：``str``


containerStorageMountPoint
--------------------------

Docker 容器中存储服务（通常是NFS）的挂载点。

类型：``str``

这必须是绝对路径。


reuseMode
---------

启动 `重用模式 <../Tutorial/ExperimentConfig.rst#reuse>`__。

类型：``bool``

默认值：``false``


openpaiConfig
-------------

嵌入的 OpenPAI 配置文件。

类型：``Optional[JSON]``


openpaiConfigFile
-----------------

到 OpenPAI 配置文件的 `路径`_

类型：``Optional[str]``

示例在 `这里 <https://github.com/microsoft/pai/blob/master/docs/manual/cluster-user/examples/hello-world-job.yaml>`__。


AmlConfig
^^^^^^^^^

详情查看 `这里 <../TrainingService/AMLMode.rst>`__。


platform
--------

字符串常量 ``"aml"``。


dockerImage
-----------

运行 Trial 的 Docker 镜像的名称和标签。

类型：``str``

默认：``"msranni/nni:latest"``


subscriptionId
--------------

Azure 订阅 ID。

类型：``str``


resourceGroup
-------------

Azure 资源组名称。

类型：``str``


workspaceName
-------------

Azure 工作区名称。

类型：``str``


computeTarget
-------------

AML 计算集群名称。

类型：``str``
