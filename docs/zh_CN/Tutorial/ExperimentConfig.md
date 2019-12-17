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
    - [multiPhase](#multiphase)
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
  - [样例](#examples) 
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
multiPhase:
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
multiPhase:
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
multiPhase:
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

可选。 布尔。 默认值：false。

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

### multiPhase

可选。 布尔。 默认值：false。

启用[多阶段 Experiment](../AdvancedFeature/MultiPhase.md)。

### multiThread

可选。 布尔。 默认值：false。

为 Dispatcher 启用多线程模式。 如果启用了 multiThread，Dispatcher 将启动一个线程来处理来自 NNI 管理器的每个命令。

### nniManagerIp

可选。 字符串。 默认值：eth0 设备的 IP。

设置运行 NNI 管理器进程的计算机的 IP 地址。 此字段为可选项，如果没有设置，则会使用 eth0 的 IP 地址。

注意: 可在 NNI 管理器机器上运行 `ifconfig` 来检查 eth0 是否存在。 如果没有，建议显式设置 **nniManagerIp**。

### logDir

可选。 目录的路径。 默认值：`<user home directory>/nni/experiment`。

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

If **includeIntermediateResults** is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result.

### assessor

Specifies the assessor algorithm to run an experiment. Similar to tuners, there are two kinds of ways to set assessor. One way is to use assessor provided by NNI sdk. Users need to set **builtinAssessorName** and **classArgs**. Another way is to use users' own assessor file, and users need to set **codeDirectory**, **classFileName**, **className** and **classArgs**. *Users must choose exactly one way.*

By default, there is no assessor enabled.

#### builtinAssessorName

Required if using built-in assessors. String.

Specifies the name of built-in assessor, NNI sdk provides different assessors introduced [here](../Assessor/BuiltinAssessor.md).

#### codeDir

Required if using customized assessors. Path relative to the location of config file.

Specifies the directory of assessor code.

#### classFileName

Required if using customized assessors. File path relative to **codeDir**.

Specifies the name of assessor file.

#### className

Required if using customized assessors. String.

Specifies the name of assessor class.

#### classArgs

Optional. Key-value pairs. Default: empty.

Specifies the arguments of assessor algorithm.

### advisor

Optional.

Specifies the advisor algorithm in the experiment. Similar to tuners and assessors, there are two kinds of ways to specify advisor. One way is to use advisor provided by NNI sdk, need to set **builtinAdvisorName** and **classArgs**. Another way is to use users' own advisor file, and need to set **codeDirectory**, **classFileName**, **className** and **classArgs**.

When advisor is enabled, settings of tuners and advisors will be bypassed.

#### builtinAdvisorName

Specifies the name of a built-in advisor. NNI sdk provides [BOHB](../Tuner/BohbAdvisor.md) and [Hyperband](../Tuner/HyperbandAdvisor.md).

#### codeDir

Required if using customized advisors. Path relative to the location of config file.

Specifies the directory of advisor code.

#### classFileName

Required if using customized advisors. File path relative to **codeDir**.

Specifies the name of advisor file.

#### className

Required if using customized advisors. String.

Specifies the name of advisor class.

#### classArgs

Optional. Key-value pairs. Default: empty.

Specifies the arguments of advisor.

#### gpuIndices

Optional. String. Default: empty.

Specifies the GPUs that can be used. Single or multiple GPU indices can be specified. Multiple GPU indices are separated by comma `,`. For example, `1`, or `0,1,3`. If the field is not set, no GPU will be visible to tuner (by setting `CUDA_VISIBLE_DEVICES` to be an empty string).

### trial

Required. Key-value pairs.

In local and remote mode, the following keys are required.

- **command**: Required string. Specifies the command to run trial process.

- **codeDir**: Required string. Specifies the directory of your own trial file. This directory will be automatically uploaded in remote mode.

- **gpuNum**: Optional integer. Specifies the num of gpu to run the trial process. Default value is 0.

In PAI mode, the following keys are required.

- **command**: Required string. Specifies the command to run trial process.

- **codeDir**: Required string. Specifies the directory of the own trial file. Files in the directory will be uploaded in PAI mode.

- **gpuNum**: Required integer. Specifies the num of gpu to run the trial process. Default value is 0.

- **cpuNum**: Required integer. Specifies the cpu number of cpu to be used in pai container.

- **memoryMB**: Required integer. Set the memory size to be used in pai container, in megabytes.

- **image**: Required string. Set the image to be used in pai.

- **authFile**: Optional string. Used to provide Docker registry which needs authentication for image pull in PAI. [Reference](https://github.com/microsoft/pai/blob/2ea69b45faa018662bc164ed7733f6fdbb4c42b3/docs/faq.md#q-how-to-use-private-docker-registry-job-image-when-submitting-an-openpai-job).

- **shmMB**: Optional integer. Shared memory size of container.

- **portList**: List of key-values pairs with `label`, `beginAt`, `portNumber`. See [job tutorial of PAI](https://github.com/microsoft/pai/blob/master/docs/job_tutorial.md) for details.

In Kubeflow mode, the following keys are required.

- **codeDir**: The local directory where the code files are in.

- **ps**: An optional configuration for kubeflow's tensorflow-operator, which includes
  
      * __replicas__: The replica number of __ps__ role.
      
      * __command__: The run script in __ps__'s container.
      
      * __gpuNum__: The gpu number to be used in __ps__ container.
      
      * __cpuNum__: The cpu number to be used in __ps__ container.
      
      * __memoryMB__: The memory size of the container.
      
      * __image__: The image to be used in __ps__.
      

- **worker**: An optional configuration for kubeflow's tensorflow-operator.
  
      * __replicas__: The replica number of __worker__ role.
      
      * __command__: The run script in __worker__'s container.
      
      * __gpuNum__: The gpu number to be used in __worker__ container.
      
      * __cpuNum__: The cpu number to be used in __worker__ container.
      
      * __memoryMB__: The memory size of the container.
      
      * __image__: The image to be used in __worker__.
      

### localConfig

Optional in local mode. Key-value pairs.

Only applicable if **trainingServicePlatform** is set to `local`, otherwise there should not be **localConfig** section in configuration file.

#### gpuIndices

Optional. String. Default: none.

Used to specify designated GPU devices for NNI, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified. Multiple GPU indices should be separated with comma (`,`), such as `1` or `0,1,3`. By default, all GPUs available will be used.

#### maxTrialNumPerGpu

Optional. Integer. Default: 99999.

Used to specify the max concurrency trial number on a GPU device.

#### useActiveGpu

Optional. Bool. Default: false.

Used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no other active process in the GPU. If **useActiveGpu** is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.

### machineList

Required in remote mode. A list of key-value pairs with the following keys.

#### ip

Required. IP address that is accessible from the current machine.

The IP address of remote machine.

#### port

Optional. Integer. Valid port. Default: 22.

The ssh port to be used to connect machine.

#### username

Required if authentication with username/password. String.

The account of remote machine.

#### passwd

Required if authentication with username/password. String.

Specifies the password of the account.

#### sshKeyPath

Required if authentication with ssh key. Path to private key file.

If users use ssh key to login remote machine, **sshKeyPath** should be a valid path to a ssh key file.

*Note: if users set passwd and sshKeyPath simultaneously, NNI will try passwd first.*

#### passphrase

Optional. String.

Used to protect ssh key, which could be empty if users don't have passphrase.

#### gpuIndices

Optional. String. Default: none.

Used to specify designated GPU devices for NNI, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified. Multiple GPU indices should be separated with comma (`,`), such as `1` or `0,1,3`. By default, all GPUs available will be used.

#### maxTrialNumPerGpu

Optional. Integer. Default: 99999.

Used to specify the max concurrency trial number on a GPU device.

#### useActiveGpu

Optional. Bool. Default: false.

Used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no other active process in the GPU. If **useActiveGpu** is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.

### kubeflowConfig

#### operator

Required. String. Has to be `tf-operator` or `pytorch-operator`.

Specifies the kubeflow's operator to be used, NNI support `tf-operator` in current version.

#### storage

Optional. String. Default. `nfs`.

Specifies the storage type of kubeflow, including `nfs` and `azureStorage`.

#### nfs

Required if using nfs. Key-value pairs.

- **server** is the host of nfs server.

- **path** is the mounted path of nfs.

#### keyVault

Required if using azure storage. Key-value pairs.

Set **keyVault** to storage the private key of your azure storage account. Refer to https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2.

- **vaultName** is the value of `--vault-name` used in az command.

- **name** is the value of `--name` used in az command.

#### azureStorage

Required if using azure storage. Key-value pairs.

Set azure storage account to store code files.

- **accountName** is the name of azure storage account.

- **azureShare** is the share of the azure file storage.

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

## 样例

### Local mode

If users want to run trial jobs in local machine, and use annotation to generate search space, could use the following config:

    ```yaml
    authorName: test
    experimentName: test_experiment
    trialConcurrency: 3
    maxExecDuration: 1h
    maxTrialNum: 10
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: local
    #choice: true, false
    useAnnotation: true
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
    ```
    

You can add assessor configuration.

    ```yaml
    authorName: test
    experimentName: test_experiment
    trialConcurrency: 3
    maxExecDuration: 1h
    maxTrialNum: 10
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: local
    searchSpacePath: /nni/search_space.json
    #choice: true, false
    useAnnotation: false
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
        optimize_mode: maximize
    assessor:
      #choice: Medianstop
      builtinAssessorName: Medianstop
      classArgs:
        #choice: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 mnist.py
      codeDir: /nni/mnist
      gpuNum: 0
    ```
    

Or you could specify your own tuner and assessor file as following,

    ```yaml
    authorName: test
    experimentName: test_experiment
    trialConcurrency: 3
    maxExecDuration: 1h
    maxTrialNum: 10
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: local
    searchSpacePath: /nni/search_space.json
    #choice: true, false
    useAnnotation: false
    tuner:
      codeDir: /nni/tuner
      classFileName: mytuner.py
      className: MyTuner
      classArgs:
        #choice: maximize, minimize
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
    ```
    

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
    ```
    

### PAI mode

    ```yaml
    authorName: test
    experimentName: nni_test1
    trialConcurrency: 1
    maxExecDuration:500h
    maxTrialNum: 1
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: pai
    searchSpacePath: search_space.json
    #choice: true, false
    useAnnotation: false
    tuner:
      #choice: TPE, Random, Anneal, Evolution, BatchTuner
      #SMAC (SMAC should be installed through nnictl)
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
        optimize_mode: maximize
    trial:
      command: python3 main.py
      codeDir: .
      gpuNum: 4
      cpuNum: 2
      memoryMB: 10000
      #The docker image to run NNI job on pai
      image: msranni/nni:latest
    paiConfig:
      #The username to login pai
      userName: test
      #The password to login pai
      passWord: test
      #The host of restful server of pai
      host: 10.10.10.10
    ```
    

### Kubeflow mode

    kubeflow with nfs storage.
    
    ```yaml
    authorName: default
    experimentName: example_mni
    trialConcurrency: 1
    maxExecDuration: 1h
    maxTrialNum: 1
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: kubeflow
    searchSpacePath: search_space.json
    #choice: true, false
    useAnnotation: false
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
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
    ```
    

### Kubeflow with azure storage

    ```yaml
    authorName: default
    experimentName: example_mni
    trialConcurrency: 1
    maxExecDuration: 1h
    maxTrialNum: 1
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: kubeflow
    searchSpacePath: search_space.json
    #choice: true, false
    useAnnotation: false
    #nniManagerIp: 10.10.10.10
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        #choice: maximize, minimize
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
    ```