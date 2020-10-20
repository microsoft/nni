# Experiment（实验）配置参考

创建 Experiment 所需要的配置文件。 配置文件的路径会传入 `nnictl` 命令。 配置文件的格式为 YAML。 本文介绍了配置文件的内容，并提供了一些示例和模板。

- [Experiment（实验）配置参考](#experiment-config-reference) 
  - [模板](#template)
  - [说明](#configuration-spec) 
    - [authorName](#authorname)
    - [experimentName](#experimentname)
    - [trialConcurrency](#trialconcurrency)
    - [maxExecDuration](#maxexecduration)
    - [versionCheck](#versioncheck)
    - [debug](#debug)
    - [maxTrialNum](#maxtrialnum)
    - [trainingServicePlatform](#trainingserviceplatform)
    - [searchSpacePath](#searchspacepath)
    - [useAnnotation](#useannotation)
    - [multiThread](#multithread)
    - [nniManagerIp](#nnimanagerip)
    - [logDir](#logdir)
    - [logLevel](#loglevel)
    - [logCollection](#logcollection)
    - [tuner](#tuner) 
      - [builtinTunerName](#builtintunername)
      - [codeDir](#codedir)
      - [classFileName](#classfilename)
      - [className](#classname)
      - [classArgs](#classargs)
      - [gpuIndices](#gpuindices)
      - [includeIntermediateResults](#includeintermediateresults)
    - [assessor](#assessor) 
      - [builtinAssessorName](#builtinassessorname)
      - [codeDir](#codedir-1)
      - [classFileName](#classfilename-1)
      - [className](#classname-1)
      - [classArgs](#classargs-1)
    - [advisor](#advisor) 
      - [builtinAdvisorName](#builtinadvisorname)
      - [codeDir](#codedir-2)
      - [classFileName](#classfilename-2)
      - [className](#classname-2)
      - [classArgs](#classargs-2)
      - [gpuIndices](#gpuindices-1)
    - [trial](#trial)
    - [localConfig](#localconfig) 
      - [gpuIndices](#gpuindices-2)
      - [maxTrialNumPerGpu](#maxtrialnumpergpu)
      - [useActiveGpu](#useactivegpu)
    - [machineList](#machinelist) 
      - [ip](#ip)
      - [port](#port)
      - [username](#username)
      - [passwd](#passwd)
      - [sshKeyPath](#sshkeypath)
      - [passphrase](#passphrase)
      - [gpuIndices](#gpuindices-3)
      - [maxTrialNumPerGpu](#maxtrialnumpergpu-1)
      - [useActiveGpu](#useactivegpu-1)
      - [preCommand](#preCommand)
    - [kubeflowConfig](#kubeflowconfig) 
      - [operator](#operator)
      - [storage](#storage)
      - [nfs](#nfs)
      - [keyVault](#keyvault)
      - [azureStorage](#azurestorage)
      - [uploadRetryCount](#uploadretrycount)
    - [paiConfig](#paiconfig) 
      - [userName](#username)
      - [password](#password)
      - [token](#token)
      - [host](#host)
      - [reuse](#reuse)
  - [示例](#examples) 
    - [本机模式](#local-mode)
    - [远程模式](#remote-mode)
    - [PAI 模式](#pai-mode)
    - [Kubeflow 模式](#kubeflow-mode)
    - [Kubeflow 中使用 Azure 存储](#kubeflow-with-azure-storage)

## 模板

- **简化版（不包含 Annotation（标记）和 Assessor）**

```yaml
authorName:
experimentName:
trialConcurrency:
maxExecDuration:
maxTrialNum:
# 可选项: local, remote, pai, kubeflow
trainingServicePlatform:
searchSpacePath:
# 可选项: true, false, 默认值: false
useAnnotation:
# 可选项: true, false, 默认值: false
multiThread:
tuner:
  # 可选项: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    # 可选项: maximize, minimize
    optimize_mode:
  gpuIndices:
trial:
  command:
  codeDir:
  gpuNum:
# 在本机模式下，machineList 可为空
machineList:
  - ip:
    port:
    username:
    passwd:
```

- **使用 Assessor**

```yaml
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#可选项: local, remote, pai, kubeflow
trainingServicePlatform: 
searchSpacePath: 
#可选项: true, false, 默认值: false
useAnnotation:
#可选项: true, false, 默认值: false
multiThread:
tuner:
  #可选项: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuIndices: 
assessor:
  #可选项: Medianstop
  builtinAssessorName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuIndices: 
trial:
  command: 
  codeDir: 
  gpuIndices: 
#在本地使用时，machineList 可为空
machineList:
  - ip: 
    port: 
    username: 
    passwd:
```

- **使用 Annotation**

```yaml
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#可选项: local, remote, pai, kubeflow
trainingServicePlatform: 
#可选项: true, false, 默认值: false
useAnnotation:
#可选项: true, false, 默认值: false
multiThread:
tuner:
  #可选项: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuIndices: 
assessor:
  #可选项: Medianstop
  builtinAssessorName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuIndices: 
trial:
  command: 
  codeDir: 
  gpuIndices: 
#在本地使用时，machineList 可为空
machineList:
  - ip: 
    port: 
    username: 
    passwd:
```

## 说明

### authorName

必填。 字符串。

创建 Experiment 的作者的姓名。

*待定: 增加默认值。*

### experimentName

必填。 字符串。

创建的 Experiment 名称。

*待定: 增加默认值。*

### trialConcurrency

必填。 1 到 99999 之间的整数。

指定同时运行的 Trial 任务的最大数量。

如果 trialGpuNum 大于空闲的 GPU 数量，并且并发的 Trial 任务数量还没达到 **trialConcurrency**，Trial 任务会被放入队列，等待分配 GPU 资源。

### maxExecDuration

可选。 字符串。 默认值：999d。

**maxExecDuration** 指定实验的最大执行时间。 时间的单位为 {**s**, **m**, **h**, **d**}，其分别表示 {*秒*, *分钟*, *小时*, *天*}。

注意：maxExecDuration 设置的是 Experiment 执行的时间，不是 Trial 的。 如果 Experiment 达到了设置的最大时间，Experiment 不会停止，但不会再启动新的 Trial 作业。

### versionCheck

可选。 布尔。 默认值：true。

NNI 会校验 remote, pai 和 Kubernetes 模式下 NNIManager 与 trialKeeper 进程的版本。 如果需要禁用版本校验，versionCheck 应设置为 false。

### debug

可选。 布尔。 默认值：false。

调试模式会将 versionCheck 设置为 False，并将 logLevel 设置为 'debug'。

### maxTrialNum

可选。 1 到 99999 之间的整数。 默认值：99999。

指定 NNI 创建的最大 Trial 任务数，包括成功和失败的任务。

### trainingServicePlatform

必填。 字符串。

指定运行 Experiment 的平台，包括 **local**, **remote**, **pai**, **kubeflow**, **frameworkcontroller**.

- **local** 在本机的 Ubuntu 上运行 Experiment。

- **remote** 将任务提交到远程的 Ubuntu 上，必须用 **machineList** 来指定远程的 SSH 连接信息。

- **pai** 提交任务到微软开源的 [OpenPAI](https://github.com/Microsoft/pai) 上。 更多 OpenPAI 配置，参考 [PAI 模式](../TrainingService/PaiMode.md)。

- **kubeflow** 提交任务至 [Kubeflow](https://www.kubeflow.org/docs/about/kubeflow/)。 NNI 支持基于 Kubeflow 的 Kubenetes，以及[Azure Kubernetes](https://azure.microsoft.com/en-us/services/kubernetes-service/)。 详情参考 [Kubeflow 文档](../TrainingService/KubeflowMode.md)

- TODO：解释 FrameworkController。

### searchSpacePath

可选。 现有文件的路径。

指定搜索空间文件的路径，此文件必须在运行 nnictl 的本机。

仅在 `useAnnotation=True` 时，才不需要填写 **searchSpacePath**。

### useAnnotation

可选。 布尔。 默认值：false。

使用 Annotation 分析 Trial 代码并生成搜索空间。

注意：如果 **useAnnotation** 为 true，searchSpacePath 字段会被删除。

### multiThread

可选。 布尔。 默认值：false。

为 Dispatcher 启用多线程模式。 如果启用了 multiThread，Dispatcher 将启动一个线程来处理来自 NNI 管理器的每个命令。

### nniManagerIp

可选。 字符串。 默认值：eth0 设备的 IP。

设置运行 NNI 管理器进程的计算机的 IP 地址。 此字段为可选项，如果没有设置，则会使用 eth0 的 IP 地址。

注意: 可在 NNI 管理器机器上运行 `ifconfig` 来检查 eth0 是否存在。 如果没有，建议显式设置 **nniManagerIp**。

### logDir

可选。 目录的路径。 默认值：`<user home directory>/nni-experiments`。

配置目录以存储 Experiment 的日志和数据。

### logLevel

可选。 字符串。 默认值：`info`。

设置 Experiment 的日志级别。 可设置的日志级别包括：`trace`, `debug`, `info`, `warning`, `error`, `fatal`。

### logCollection

可选。 `http` 或 `none`。 默认值：`none`。

设置在remote、pai、kubeflow、frameworkcontroller 平台中收集日志的方式。 日志支持两种设置，一种是通过 `http`，让 Trial 将日志通过 POST 方法发回日志，这种方法会减慢 trialKeeper 的速度。 另一种方法是 `none`，Trial 不将日志回传回来，仅仅回传 Job 的指标。 如果日志较大，可将此参数设置为 `none`。

### tuner

必填。

指定了 Experiment 的 Tuner 算法。有两种方法可设置 Tuner。 一种方法是使用 NNI SDK 提供的内置 Tuner，在这种情况下，需要设置 **builtinTunerName** 和 **classArgs**。 另一种方法，是使用用户自定义的 Tuner，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。 *必须选择其中的一种方式。*

#### builtinTunerName

如果使用内置 Tuner，则为必需。 字符串。

指定系统 Tuner 的名称, NNI SDK 提供的各种 Tuner 的[说明](../Tuner/BuiltinTuner.md)。

#### codeDir

如果使用定制 Tuner，则为必需。 相对于配置文件位置的路径。

指定 Tuner 代码的目录。

#### classFileName

如果使用定制 Tuner，则为必需。 相对于 **codeDir** 的文件路径。

指定 Tuner 文件的名称。

#### className

如果使用定制 Tuner，则为必需。 字符串。

指定 Tuner 的名称。

#### classArgs

可选。 键值对。 默认值：空。

指定 Tuner 算法的参数。 参考[此文件](../Tuner/BuiltinTuner.md)来了解内置 Tuner 的配置参数。

#### gpuIndices

可选。 字符串。 默认值：空。

指定 Tuner 进程可以使用的 GPU。 可以指定单个或多个 GPU 索引。 多个 GPU 索引用逗号 `,` 分隔。 例如，`1` 或 `0,1,3`。 如果未设置该字段，则 Tuner 将找不到 GPU（设置 `CUDA_VISIBLE_DEVICES` 成空字符串）。

#### includeIntermediateResults

可选。 布尔。 默认值：false。

如果 **includeIntermediateResults** 为 true，最后一个 Assessor 的中间结果会被发送给 Tuner 作为最终结果。

### assessor

指定 Assessor 算法以运行 Experiment。 与 Tuner 类似，有两种设置 Assessor 的方法。 一种方法是使用 NNI SDK 提供的 Assessor。 用户需要设置 **builtinAssessorName** 和 **classArgs**。 另一种方法，是使用自定义的 Assessor，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。 *必须选择其中的一种方式。*

默认情况下，未启用任何 Assessor。

#### builtinAssessorName

如果使用内置 Assessor，则为必需。 字符串。

指定内置 Assessor 的名称，NNI SDK 提供的 Assessor 可参考[这里](../Assessor/BuiltinAssessor.md)。

#### codeDir

如果使用定制 Assessor，则为必需。 相对于配置文件位置的路径。

指定 Assessor 代码的目录。

#### classFileName

如果使用定制 Assessor，则为必需。 相对于 **codeDir** 的文件路径。

指定 Assessor 文件的名称。

#### className

如果使用定制 Assessor，则为必需。 字符串。

指定 Assessor 类的名称。

#### classArgs

可选。 键值对。 默认值：空。

指定 Assessor 算法的参数。

### advisor

可选。

指定 Experiment 中的 Advisor 算法。 与 Tuner 和 Assessor 类似，有两种指定 Advisor 的方法。 一种方法是使用 SDK 提供的 Advisor ，需要设置 **builtinAdvisorName** 和 **classArgs**。 另一种方法，是使用用户自定义的 Advisor，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。

启用 Advisor 后，将忽略 Tuner 和 Advisor 的设置。

#### builtinAdvisorName

指定内置 Advisor 的名称。 NNI SDK 提供了 [BOHB](../Tuner/BohbAdvisor.md) 和 [Hyperband](../Tuner/HyperbandAdvisor.md)。

#### codeDir

如果使用定制 Advisor，则为必需。 相对于配置文件位置的路径。

指定 Advisor 代码的目录。

#### classFileName

如果使用定制 Advisor，则为必需。 相对于 **codeDir** 的文件路径。

指定 Advisor 文件的名称。

#### className

如果使用定制 Advisor，则为必需。 字符串。

指定 Advisor 类的名称。

#### classArgs

可选。 键值对。 默认值：空。

指定 Advisor 的参数。

#### gpuIndices

可选。 字符串。 默认值：空。

指定可以使用的 GPU。 可以指定单个或多个 GPU 索引。 多个 GPU 索引用逗号 `,` 分隔。 例如，`1` 或 `0,1,3`。 如果未设置该字段，则 Tuner 将找不到 GPU（设置 `CUDA_VISIBLE_DEVICES` 成空字符串）。

### trial

必填。 键值对。

在 local 和 remote 模式下，需要以下键。

- **command**：必需字符串。 指定运行 Trial 的命令。

- **codeDir**：必需字符串。 指定 Trial 文件的目录。 此目录将在 remote 模式下自动上传。

- **gpuNum**：可选、整数。 指定了运行 Trial 进程的 GPU 数量。 默认值为 0。

在 PAI 模式下，需要以下键。

- **command**：必需字符串。 指定运行 Trial 的命令。

- **codeDir**：必需字符串。 指定 Trial 文件的目录。 目录中的文件将在 PAI 模式下上传。

- **gpuNum**：必需、整数。 指定了运行 Trial 进程的 GPU 数量。 默认值为 0。

- **cpuNum**：必需、整数。 指定要在 OpenPAI 容器中使用的 cpu 数。

- **memoryMB**：必需、整数。 设置要在 OpenPAI 容器中使用的内存大小，以兆字节为单位。

- **image**：必需字符串。 设置要在 OpenPAI 中使用的 Docker 映像。

- **authFile**：可选、字符串。 用于提供 Docker 注册，用于为 OpenPAI 中的映像拉取请求进行身份验证。 [参考](https://github.com/microsoft/pai/blob/2ea69b45faa018662bc164ed7733f6fdbb4c42b3/docs/faq.md#q-how-to-use-private-docker-registry-job-image-when-submitting-an-openpai-job)。

- **shmMB**：可选、整数。 容器的共享内存大小。

- **portList**: `label`, `beginAt`, `portNumber` 的键值对 list。 参考[ OpenPAI Job 教程](https://github.com/microsoft/pai/blob/master/docs/job_tutorial.md)。

在 Kubeflow 模式下，需要以下键。

- **codeDir** 指定了代码文件的本机路径。

- **ps**: Kubeflow 的 tensorflow-operator 的可选配置，包括：
  
      * __replicas__: __ps__ 角色的副本数量。
      
      * __command__: __ps__ 容器的运行脚本。
      
      * __gpuNum__: 在 __ps__ 容器中使用的 GPU 数量。
      
      * __cpuNum__: 在 __ps__ 容器中使用的 CPU 数量。
      
      * __memoryMB__：容器的内存大小。
      
      * __image__: 在 __ps__ 中使用的 Docker 映像。
      

- **worker** 是 Kubeflow 的 tensorflow-operator 的可选配置。
  
      * __replicas__: __worker__ 角色的副本数量。
      
      * __command__: __worker__ 容器的运行脚本。
      
      * __gpuNum__: 在 __worker__ 容器中使用的 GPU 数量。
      
      * __cpuNum__: 在 __worker__ 容器中使用的 CPU 数量。
      
      * __memoryMB__：容器的内存大小。
      
      * __image__: 在 __worker__ 中使用的 Docker 映像。
      

### localConfig

本机模式下可选。 键值对。

仅在 **trainingServicePlatform** 设为 `local` 时有效，否则，配置文件中不应该有 **localConfig** 部分。

#### gpuIndices

可选。 字符串。 默认值：none。

用于指定特定的 GPU。设置此值后，只有指定的 GPU 会被用来运行 Trial 任务。 可以指定单个或多个 GPU 索引。 多个 GPU 索引，应用逗号（`,`）分隔，如 `1` 或 `0,1,3`。 默认情况下，将使用所有可用的 GPU。

#### maxTrialNumPerGpu

可选。 整数。 默认值： 1。

用于指定 GPU 设备上的最大并发 Trial 的数量。

#### useActiveGpu

可选。 布尔。 默认值：false。

用于指定 GPU 上存在其他进程时是否使用此 GPU。 默认情况下，NNI 仅在 GPU 中没有其他活动进程时才使用 GPU。 如果 **useActiveGpu** 设置为 true，则 NNI 无论某 GPU 是否有其它进程，都将使用它。 此字段不适用于 Windows 版的 NNI。

### machineList

在 remote 模式下必需。 具有以下键的键值对的列表。

#### ip

必填。 可从当前计算机访问的 IP 地址或主机名。

远程计算机的 IP 地址或主机名。

#### port

可选。 整数。 有效端口。 默认值： 22。

用于连接计算机的 SSH 端口。

#### username

使用用户名/密码进行身份验证时是必需的。 字符串。

远程计算机的帐户。

#### passwd

使用用户名/密码进行身份验证时是必需的。 字符串。

指定帐户的密码。

#### sshKeyPath

如果使用 SSH 密钥进行身份验证，则为必需。 私钥文件的路径。

如果用户使用 SSH 密钥登录远程计算机，**sshKeyPath** 应是有效的 SSH 密钥文件路径。

*注意：如果同时设置了 passwd 和 sshKeyPath，NNI 会首先使用 passwd。*

#### passphrase

可选。 字符串。

用于保护 SSH 密钥，如果用户没有密码，可为空。

#### gpuIndices

可选。 字符串。 默认值：none。

用于指定特定的 GPU。设置此值后，只有指定的 GPU 会被用来运行 Trial 任务。 可以指定单个或多个 GPU 索引。 多个 GPU 索引，应用逗号（`,`）分隔，如 `1` 或 `0,1,3`。 默认情况下，将使用所有可用的 GPU。

#### maxTrialNumPerGpu

可选。 整数。 默认值： 1。

用于指定 GPU 设备上的最大并发 Trial 的数量。

#### useActiveGpu

可选。 布尔。 默认值：false。

用于指定 GPU 上存在其他进程时是否使用此 GPU。 默认情况下，NNI 仅在 GPU 中没有其他活动进程时才使用 GPU。 如果 **useActiveGpu** 设置为 true，则 NNI 无论某 GPU 是否有其它进程，都将使用它。 此字段不适用于 Windows 版的 NNI。

#### preCommand

Optional. 字符串。

Specifies the pre-command that will be executed before the remote machine executes other commands. Users can configure the experimental environment on remote machine by setting **preCommand**. If there are multiple commands need to execute, use `&&` to connect them, such as `preCommand: command1 && command2 && ...`.

**Note**: Because **preCommand** will execute before other commands each time, it is strongly not recommended to set **preCommand** that will make changes to system, i.e. `mkdir` or `touch`.

### remoteConfig

Optional field in remote mode. Users could set per machine information in `machineList` field, and set global configuration for remote mode in this field.

#### reuse

Optional. Bool. default: `false`. It's an experimental feature.

If it's true, NNI will reuse remote jobs to run as many as possible trials. It can save time of creating new jobs. User needs to make sure each trial can run independent in same job, for example, avoid loading checkpoint from previous trials.

### kubeflowConfig

#### operator

Required. String. Has to be `tf-operator` or `pytorch-operator`.

Specifies the kubeflow's operator to be used, NNI support `tf-operator` in current version.

#### storage

Optional. String. Default. `nfs`.

Specifies the storage type of kubeflow, including `nfs` and `azureStorage`.

#### nfs

Required if using nfs. Key-value pairs.

- **server** 是 NFS 服务器的地址。

- **path** 是 NFS 挂载的路径。

#### keyVault

Required if using azure storage. Key-value pairs.

Set **keyVault** to storage the private key of your azure storage account. Refer to https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2.

- **vaultName** 是 az 命令中 `--vault-name` 的值。

- **name** 是 az 命令中 `--name` 的值。

#### azureStorage

Required if using azure storage. Key-value pairs.

Set azure storage account to store code files.

- **accountName** 是 Azure 存储账户的名称。

- **azureShare** 是 Azure 文件存储的共享参数。

#### uploadRetryCount

Required if using azure storage. Integer between 1 and 99999.

If upload files to azure storage failed, NNI will retry the process of uploading, this field will specify the number of attempts to re-upload files.

### paiConfig

#### userName

Required. String.

The user name of your pai account.

#### password

Required if using password authentication. String.

The password of the pai account.

#### token

Required if using token authentication. String.

Personal access token that can be retrieved from PAI portal.

#### host

Required. String.

The hostname of IP address of PAI.

#### reuse

Optional. Bool. default: `false`. It's an experimental feature.

If it's true, NNI will reuse OpenPAI jobs to run as many as possible trials. It can save time of creating new jobs. User needs to make sure each trial can run independent in same job, for example, avoid loading checkpoint from previous trials.

## 示例

### Local mode

If users want to run trial jobs in local machine, and use annotation to generate search space, could use the following config:

    authorName: test
    experimentName: test_experiment
    trialConcurrency: 3
    maxExecDuration: 1h
    maxTrialNum: 10
    #可选项: local, remote, pai, kubeflow
    trainingServicePlatform: local
    #可选项: true, false
    useAnnotation: true
    tuner:
      #可选项: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        #可选项: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 mnist.py
      codeDir: /nni/mnist
      gpuNum: 0
    

You can add assessor configuration.

    authorName: test
    experimentName: test_experiment
    trialConcurrency: 3
    maxExecDuration: 1h
    maxTrialNum: 10
    #可选项: local, remote, pai, kubeflow
    trainingServicePlatform: local
    searchSpacePath: /nni/search_space.json
    #可选项: true, false
    useAnnotation: false
    tuner:
      #可选项: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        #可选项: maximize, minimize
        optimize_mode: maximize
    assessor:
      #可选项: Medianstop
      builtinAssessorName: Medianstop
      classArgs:
        #可选项: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 mnist.py
      codeDir: /nni/mnist
      gpuNum: 0
    

Or you could specify your own tuner and assessor file as following,

    authorName: test
    experimentName: test_experiment
    trialConcurrency: 3
    maxExecDuration: 1h
    maxTrialNum: 10
    #可选项: local, remote, pai, kubeflow
    trainingServicePlatform: local
    searchSpacePath: /nni/search_space.json
    #可选项: true, false
    useAnnotation: false
    tuner:
      codeDir: /nni/tuner
      classFileName: mytuner.py
      className: MyTuner
      classArgs:
        #可选项: maximize, minimize
        optimize_mode: maximize
    assessor:
      codeDir: /nni/assessor
      classFileName: myassessor.py
      className: MyAssessor
      classArgs:
        #choice: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 mnist.py
      codeDir: /nni/mnist
      gpuNum: 0
    

### Remote mode

If run trial jobs in remote machine, users could specify the remote machine information as following format:

    ```yaml
    authorName: test
    experimentName: test_experiment
    trialConcurrency: 3
    maxExecDuration: 1h
    maxTrialNum: 10
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: remote
    searchSpacePath: /nni/search_space.json
    #choice: true, false
    useAnnotation: false
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 mnist.py
      codeDir: /nni/mnist
      gpuNum: 0
    #machineList can be empty if the platform is local
    machineList:
      - ip: 10.10.10.10
        port: 22
        username: test
        passwd: test
      - ip: 10.10.10.11
        port: 22
        username: test
        passwd: test
      - ip: 10.10.10.12
        port: 22
        username: test
        sshKeyPath: /nni/sshkey
        passphrase: qwert
        # Pre-command will be executed before the remote machine executes other commands.
        # Below is an example of specifying python environment.
        # If you want to execute multiple commands, please use "&&" to connect them.
        # preCommand: source ${replace_to_absolute_path_recommended_here}/bin/activate
        # preCommand: source ${replace_to_conda_path}/bin/activate ${replace_to_conda_env_name}
        preCommand: export PATH=${replace_to_python_environment_path_in_your_remote_machine}:$PATH
    ```
    

### PAI mode

    authorName: test
    experimentName: nni_test1
    trialConcurrency: 1
    maxExecDuration:500h
    maxTrialNum: 1
    #可选项: local, remote, pai, kubeflow
    trainingServicePlatform: pai
    searchSpacePath: search_space.json
    #可选项: true, false
    useAnnotation: false
    tuner:
      #可选项: TPE, Random, Anneal, Evolution, BatchTuner
      #SMAC (SMAC 需要使用 nnictl package 单独安装)
      builtinTunerName: TPE
      classArgs:
        #可选项: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 main.py
      codeDir: .
      gpuNum: 4
      cpuNum: 2
      memoryMB: 10000
      # 在 OpenPAI 上运行 NNI 的 Docker 映像
      image: msranni/nni:latest
    paiConfig:
      # 登录 OpenPAI 的用户名
      userName: test
      # 登录 OpenPAI 的密码
      passWord: test
      # OpenPAI 的 RestFUL 服务器地址
      host: 10.10.10.10
    

### Kubeflow mode

    使用 NFS 存储。
    
    authorName: default
    experimentName: example_mni
    trialConcurrency: 1
    maxExecDuration: 1h
    maxTrialNum: 1
    # 可选项: local, remote, pai, kubeflow
    trainingServicePlatform: kubeflow
    searchSpacePath: search_space.json
    # 可选项: true, false
    useAnnotation: false
    tuner:
      # 可选项: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        # 可选项: maximize, minimize
        optimize_mode: maximize
    trial:
      codeDir: .
      worker:
        replicas: 1
        command: python3 mnist.py
        gpuNum: 0
        cpuNum: 1
        memoryMB: 8192
        image: msranni/nni:latest
    kubeflowConfig:
      operator: tf-operator
      nfs:
        server: 10.10.10.10
        path: /var/nfs/general
    
    

### Kubeflow with azure storage

    authorName: default
    experimentName: example_mni
    trialConcurrency: 1
    maxExecDuration: 1h
    maxTrialNum: 1
    # 可选项: local, remote, pai, kubeflow
    trainingServicePlatform: kubeflow
    searchSpacePath: search_space.json
    # 可选项: true, false
    useAnnotation: false
    #nniManagerIp: 10.10.10.10
    tuner:
      # 可选项: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        # 可选项: maximize, minimize
        optimize_mode: maximize
    assessor:
      builtinAssessorName: Medianstop
      classArgs:
        optimize_mode: maximize
    trial:
      codeDir: .
      worker:
        replicas: 1
        command: python3 mnist.py
        gpuNum: 0
        cpuNum: 1
        memoryMB: 4096
        image: msranni/nni:latest
    kubeflowConfig:
      operator: tf-operator
      keyVault:
        vaultName: Contoso-Vault
        name: AzureStorageAccountKey
      azureStorage:
        accountName: storage
        azureShare: share01