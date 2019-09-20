# Experiment（实验）配置参考

创建 Experiment 所需要的配置文件。 配置文件的路径会传入 `nnictl` 命令。 配置文件的格式为 YAML。 本文介绍了配置文件的内容，并提供了一些示例和模板。

- [Experiment（实验）配置参考](#Experiment-config-reference) 
  - [模板](#Template)
  - [说明](#Configuration-spec)
  - [样例](#Examples)

<a name="Template"></a>

## 模板

- **简化版（不包含 Annotation（标记）和 Assessor）**

```yaml
authorName:
experimentName:
trialConcurrency:
maxExecDuration:
maxTrialNum:
#choice: local, remote, pai, kubeflow
trainingServicePlatform:
searchSpacePath:
#choice: true, false, default: false
useAnnotation:
#choice: true, false, default: false
multiPhase:
#choice: true, false, default: false
multiThread:
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuIndices:
trial:
  command:
  codeDir:
  gpuNum:
#machineList can be empty if the platform is local
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
#choice: local, remote, pai, kubeflow
trainingServicePlatform:
searchSpacePath:
#choice: true, false, default: false
useAnnotation:
#choice: true, false, default: false
multiPhase:
#choice: true, false, default: false
multiThread:
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuIndices:
assessor:
  #choice: Medianstop
  builtinAssessorName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
trial:
  command:
  codeDir:
  gpuNum:
#machineList can be empty if the platform is local
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
#choice: local, remote, pai, kubeflow
trainingServicePlatform:
#choice: true, false, default: false
useAnnotation:
#choice: true, false, default: false
multiPhase:
#choice: true, false, default: false
multiThread:
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuIndices:
assessor:
  #choice: Medianstop
  builtinAssessorName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
trial:
  command:
  codeDir:
  gpuNum:
#machineList can be empty if the platform is local
machineList:
  - ip:
    port:
    username:
    passwd:
```

<a name="Configuration"></a>

## 说明

- **authorName**
  
  - 说明
    
    **authorName** 是创建 Experiment 的作者。
    
    待定: 增加默认值

- **experimentName**
  
  - 说明
    
    **experimentName** 是创建的 Experiment 的名称。
    
    待定: 增加默认值

- **trialConcurrency**
  
  - 说明
    
    **trialConcurrency** 定义了并发尝试任务的最大数量。
    
    注意：如果 trialGpuNum 大于空闲的 GPU 数量，并且并发的 Trial 任务数量还没达到 trialConcurrency，Trial 任务会被放入队列，等待分配 GPU 资源。

- **maxExecDuration**
  
  - 说明
    
    **maxExecDuration** 定义 Experiment 执行的最长时间。时间单位：{**s**, **m**, **h**, **d**}，分别代表：{*seconds*, *minutes*, *hours*, *days*}。
    
    注意：maxExecDuration 设置的是 Experiment 执行的时间，不是 Trial 的。 如果 Experiment 达到了设置的最大时间，Experiment 不会停止，但不会再启动新的 Trial 作业。

- **versionCheck**
  
  - 说明
    
    NNI 会校验 remote, pai 和 Kubernetes 模式下 NNIManager 与 trialKeeper 进程的版本。 如果需要禁用版本校验，versionCheck 应设置为 false。

- **debug**
  
  - 说明
    
    调试模式会将 versionCheck 设置为 False，并将 logLevel 设置为 'debug'。

- **maxTrialNum**
  
  - 说明
    
    **maxTrialNum** 定义了 Trial 任务的最大数量，成功和失败的都计算在内。

- **trainingServicePlatform**
  
  - 说明
    
    **trainingServicePlatform** 定义运行 Experiment 的平台，包括：{**local**, **remote**, **pai**, **kubeflow**}.
    
    - **local** 在本机的 Ubuntu 上运行 Experiment。
    
    - **remote** 将任务提交到远程的 Ubuntu 上，必须用 **machineList** 来指定远程的 SSH 连接信息。
    
    - **pai** 提交任务到微软开源的 [OpenPAI](https://github.com/Microsoft/pai) 上。 更多 OpenPAI 配置，参考 [pai 模式](../TrainingService/PaiMode.md)。
    
    - **kubeflow** 提交任务至 [Kubeflow](https://www.kubeflow.org/docs/about/kubeflow/)。 NNI 支持基于 Kubeflow 的 Kubenetes，以及[Azure Kubernetes](https://azure.microsoft.com/en-us/services/kubernetes-service/)。 详情参考 [Kubeflow 文档](../TrainingService/KubeflowMode.md)

- **searchSpacePath**
  
  - 说明
    
    **searchSpacePath** 定义搜索空间文件的路径，此文件必须在运行 nnictl 的本机。
    
    注意: 如果设置了 useAnnotation=True，searchSpacePath 字段必须被删除。

- **useAnnotation**
  
  - 说明
    
    **useAnnotation** 定义使用标记来分析代码并生成搜索空间。
    
    注意: 如果设置了 useAnnotation=True，searchSpacePath 字段必须被删除。

- **multiPhase**
  
  - 说明
    
    **multiPhase** 启用[多阶段 Experiment](../AdvancedFeature/MultiPhase.md)。

- **multiThread**
  
  - 说明
    
    **multiThread** 如果 multiThread 设为 `true`，可启动 Dispatcher 的多线程模式。Dispatcher 会为来自 NNI 管理器的每个命令启动一个线程。

- **nniManagerIp**
  
  - 说明
    
    **nniManagerIp** 设置 NNI 管理器运行的 IP 地址。 此字段为可选项，如果没有设置，则会使用 eth0 的 IP 地址。
    
    注意: 可在 NNI 管理器机器上运行 ifconfig 来检查 eth0 是否存在。 如果不存在，推荐显式设置 nnimanagerIp。

- **logDir**
  
  - 说明
    
    **logDir** 配置存储日志和数据的目录。 默认值是 `<user home directory>/nni/experiment`

- **logLevel**
  
  - 说明
    
    **logLevel** 为 Experiment 设置日志级别，支持的日志级别有：`trace, debug, info, warning, error, fatal`。 默认值是 `info`。

- **logCollection**
  
  - 说明
    
    **logCollection** 设置在 remote, pai, kubeflow, frameworkcontroller 平台下收集日志的方法。 日志支持两种设置，一种是通过 `http`，让 Trial 将日志通过 POST 方法发回日志，这种方法会减慢 trialKeeper 的速度。 另一种方法是 `none`，Trial 不将日志回传回来，仅仅回传 Job 的指标。 如果日志较大，可将此参数设置为 `none`。

- **Tuner**
  
  - 说明
    
    **tuner** 指定了 Experiment 的 Tuner 算法。有两种方法可设置 Tuner。 一种方法是使用 SDK 提供的 Tuner，需要设置 **builtinTunerName** 和 **classArgs**。 另一种方法，是使用用户自定义的 Tuner，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  - **builtinTunerName** 和 **classArgs**
    
    - **builtinTunerName**
      
      **builtinTunerName** specifies the name of system tuner, NNI sdk provides different tuners introduced [here](../Tuner/BuiltinTuner.md).
    
    - **classArgs**
      
      **classArgs** 指定了 Tuner 算法的参数。 Please refer to [this file](../Tuner/BuiltinTuner.md) for the configurable arguments of each built-in tuner.
  
  - **codeDir**, **classFileName**, **className** 和 **classArgs**
    
    - **codeDir**
      
      **codeDir** 指定 Tuner 代码的目录。
    
    - **classFileName**
      
      **classFileName** 指定 Tuner 文件名。
    
    - **className**
      
      **className** 指定 Tuner 类名。
    
    - **classArgs**
      
      **classArgs** 指定了 Tuner 算法的参数。
  
  - **gpuIndices**
    
        __gpuIndices__ specifies the gpus that can be used by the tuner process. Single or multiple GPU indices can be specified, multiple GPU indices are seperated by comma(,), such as `1` or `0,1,3`. If the field is not set, `CUDA_VISIBLE_DEVICES` will be '' in script, that is, no GPU is visible to tuner.
        
  
  - **includeIntermediateResults**
    
        如果 __includeIntermediateResults__ 为 true，最后一个 Assessor 的中间结果会被发送给 Tuner 作为最终结果。 __includeIntermediateResults__ 的默认值为 false。
        
  
  Note: users could only use one way to specify tuner, either specifying `builtinTunerName` and `classArgs`, or specifying `codeDir`, `classFileName`, `className` and `classArgs`.

- **Assessor**
  
  - 说明
    
    **assessor** 指定了 Experiment 的 Assessor 算法。有两种方法可设置 Assessor。 一种方法是使用 SDK 提供的 Assessor，需要设置 **builtinAssessorName** 和 **classArgs**。 另一种方法，是使用用户自定义的 Assessor，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  - **builtinAssessorName** 和 **classArgs**
    
    - **builtinAssessorName**
      
      **builtinAssessorName** specifies the name of built-in assessor, NNI sdk provides different assessors introducted [here](../Assessor/BuiltinAssessor.md).
    
    - **classArgs**
      
      **classArgs** 指定了 Assessor 算法的参数。
  
  - **codeDir**, **classFileName**, **className** 和 **classArgs**
    
    - **codeDir**
      
      **codeDir** 指定 Assessor 代码的目录。
    
    - **classFileName**
      
      **classFileName** 指定 Assessor 文件名。
    
    - **className**
      
      **className** 指定 Assessor 类名。
    
    - **classArgs**
      
      **classArgs** 指定了 Assessor 算法的参数。
  
  Note: users could only use one way to specify assessor, either specifying `builtinAssessorName` and `classArgs`, or specifying `codeDir`, `classFileName`, `className` and `classArgs`. If users do not want to use assessor, assessor fileld should leave to empty.

- **advisor**
  
  - Description
    
    **advisor** specifies the advisor algorithm in the experiment, there are two kinds of ways to specify advisor. One way is to use advisor provided by NNI sdk, need to set **builtinAdvisorName** and **classArgs**. Another way is to use users' own advisor file, and need to set **codeDirectory**, **classFileName**, **className** and **classArgs**.
  
  - **builtinAdvisorName** and **classArgs**
    
    - **builtinAdvisorName**
      
      **builtinAdvisorName** specifies the name of a built-in advisor, NNI sdk provides [different advisors](../Tuner/BuiltinTuner.md).
    
    - **classArgs**
      
      **classArgs** specifies the arguments of the advisor algorithm. Please refer to [this file](../Tuner/BuiltinTuner.md) for the configurable arguments of each built-in advisor.
  
  - **codeDir**, **classFileName**, **className** and **classArgs**
    
    - **codeDir**
      
      **codeDir** specifies the directory of advisor code.
    
    - **classFileName**
      
      **classFileName** specifies the name of advisor file.
    
    - **className**
      
      **className** specifies the name of advisor class.
    
    - **classArgs**
      
      **classArgs** specifies the arguments of advisor algorithm.
  
  - **gpuIndices**
    
        __gpuIndices__ specifies the gpus that can be used by the tuner process. Single or multiple GPU indices can be specified, multiple GPU indices are seperated by comma(,), such as `1` or `0,1,3`. If the field is not set, `CUDA_VISIBLE_DEVICES` will be '' in script, that is, no GPU is visible to tuner.
        
  
  Note: users could only use one way to specify advisor, either specifying `builtinAdvisorName` and `classArgs`, or specifying `codeDir`, `classFileName`, `className` and `classArgs`.

- **trial(local, remote)**
  
  - **command**
    
    **command** 指定了运行 Trial 进程的命令行。
  
  - **codeDir**
    
    **codeDir** specifies the directory of your own trial file.
  
  - **gpuNum**
    
    **gpuNum** 指定了运行 Trial 进程的 GPU 数量。 默认值为 0。

- **trial(pai)**
  
  - **command**
    
    **command** specifies the command to run trial process.
  
  - **codeDir**
    
    **codeDir** specifies the directory of the own trial file.
  
  - **gpuNum**
    
    **gpuNum** specifies the num of gpu to run the trial process. Default value is 0.
  
  - **cpuNum**
    
    **cpuNum** is the cpu number of cpu to be used in pai container.
  
  - **memoryMB**
    
    **memoryMB** set the momory size to be used in pai's container.
  
  - **image**
    
    **image** set the image to be used in pai.
  
  - **dataDir**
    
    **dataDir** is the data directory in hdfs to be used.
  
  - **outputDir**
    
    **outputDir** is the output directory in hdfs to be used in pai, the stdout and stderr files are stored in the directory after job finished.

- **trial(kubeflow)**
  
  - **codeDir**
    
    **codeDir** is the local directory where the code files in.
  
  - **ps(optional)**
    
    **ps** is the configuration for kubeflow's tensorflow-operator.
    
    - **replicas**
      
      **replicas** is the replica number of **ps** role.
    
    - **command**
      
      **command** is the run script in **ps**'s container.
    
    - **gpuNum**
      
      **gpuNum** set the gpu number to be used in **ps** container.
    
    - **cpuNum**
      
      **cpuNum** set the cpu number to be used in **ps** container.
    
    - **memoryMB**
      
      **memoryMB** set the memory size of the container.
    
    - **image**
      
      **image** set the image to be used in **ps**.
  
  - **worker**
    
    **worker** is the configuration for kubeflow's tensorflow-operator.
    
    - **replicas**
      
      **replicas** is the replica number of **worker** role.
    
    - **command**
      
      **command** is the run script in **worker**'s container.
    
    - **gpuNum**
      
      **gpuNum** set the gpu number to be used in **worker** container.
    
    - **cpuNum**
      
      **cpuNum** set the cpu number to be used in **worker** container.
    
    - **memoryMB**
      
      **memoryMB** set the memory size of the container.
    
    - **image**
      
      **image** set the image to be used in **worker**.

- **localConfig**
  
  **localConfig** is applicable only if **trainingServicePlatform** is set to `local`, otherwise there should not be **localConfig** section in configuration file.
  
  - **gpuIndices**
    
    **gpuIndices** is used to specify designated GPU devices for NNI, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified, multiple GPU indices are seperated by comma(,), such as `1` or `0,1,3`.
  
  - **maxTrialNumPerGpu**
    
    **maxTrialNumPerGpu** is used to specify the max concurrency trial number on a GPU device.
  
  - **useActiveGpu**
    
    **useActiveGpu** is used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no another active process in the GPU, if **useActiveGpu** is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.

- **machineList**
  
  **machineList** should be set if **trainingServicePlatform** is set to remote, or it should be empty.
  
  - **ip**
    
    **ip** is the ip address of remote machine.
  
  - **port**
    
    **port** is the ssh port to be used to connect machine.
    
    Note: if users set port empty, the default value will be 22.
  
  - **username**
    
    **username** is the account of remote machine.
  
  - **passwd**
    
    **passwd** specifies the password of the account.
  
  - **sshKeyPath**
    
    If users use ssh key to login remote machine, could set **sshKeyPath** in config file. **sshKeyPath** is the path of ssh key file, which should be valid.
    
    Note: if users set passwd and sshKeyPath simultaneously, NNI will try passwd.
  
  - **passphrase**
    
    **passphrase** is used to protect ssh key, which could be empty if users don't have passphrase.
  
  - **gpuIndices**
    
    **gpuIndices** is used to specify designated GPU devices for NNI on this remote machine, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified, multiple GPU indices are seperated by comma(,), such as `1` or `0,1,3`.
  
  - **maxTrialNumPerGpu**
    
    **maxTrialNumPerGpu** is used to specify the max concurrency trial number on a GPU device.
  
  - **useActiveGpu**
    
    **useActiveGpu** is used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no another active process in the GPU, if **useActiveGpu** is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.

- **kubeflowConfig**:
  
  - **operator**
    
    **operator** specify the kubeflow's operator to be used, NNI support **tf-operator** in current version.
  
  - **storage**
    
    **storage** specify the storage type of kubeflow, including {**nfs**, **azureStorage**}. This field is optional, and the default value is **nfs**. If the config use azureStorage, this field must be completed.
  
  - **nfs**
    
    **server** is the host of nfs server
    
    **path** is the mounted path of nfs
  
  - **keyVault**
    
    If users want to use azure kubernetes service, they should set keyVault to storage the private key of your azure storage account. Refer: https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2
    
    - **vaultName**
      
      **vaultName** is the value of `--vault-name` used in az command.
    
    - **name**
      
      **name** is the value of `--name` used in az command.
  
  - **azureStorage**
    
    If users use azure kubernetes service, they should set azure storage account to store code files.
    
    - **accountName**
      
      **accountName** is the name of azure storage account.
    
    - **azureShare**
      
      **azureShare** is the share of the azure file storage.
  
  - **uploadRetryCount**
    
    If upload files to azure storage failed, NNI will retry the process of uploading, this field will specify the number of attempts to re-upload files.

- **paiConfig**
  
  - **userName**
    
    **userName** is the user name of your pai account.
  
  - **password**
    
    **password** is the password of the pai account.
  
  - **host**
    
    **host** is the host of pai.

<a name="Examples"></a>

## 样例

- **本机模式**
  
  如果要在本机运行 Trial 任务，并使用标记来生成搜索空间，可参考下列配置：
  
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
  
  增加 Assessor 配置
  
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
  
  或者可以指定自定义的 Tuner 和 Assessor：
  
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

- **远程模式**
  
  如果要在远程服务器上运行 Trial 任务，需要增加服务器信息：
  
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

- **pai 模式**
  
  ```yaml
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
    # 在 OpenPAI 上用来运行 Nni 作业的 docker 映像
    image: msranni/nni:latest
    # 在 OpenPAI 的 hdfs 上存储数据的目录，如：'hdfs://host:port/directory'
    dataDir: hdfs://10.11.12.13:9000/test
    # 在 OpenPAI 的 hdfs 上存储输出的目录，如：'hdfs://host:port/directory'
    outputDir: hdfs://10.11.12.13:9000/test
  paiConfig:
    # OpenPAI 用户名
    userName: test
    # OpenPAI 密码
    passWord: test
    # OpenPAI 服务器 Ip
    host: 10.10.10.10
  ```

- **Kubeflow 模式**
  
  使用 NFS 存储。
  
  ```yaml
  authorName: default
  experimentName: example_mni
  trialConcurrency: 1
  maxExecDuration: 1h
  maxTrialNum: 1
  #可选项: local, remote, pai, kubeflow
  trainingServicePlatform: kubeflow
  searchSpacePath: search_space.json
  #可选项: true, false
  useAnnotation: false
  tuner:
    #可选项: TPE, Random, Anneal, Evolution
    builtinTunerName: TPE
    classArgs:
      #可选项: maximize, minimize
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
  
  使用 Azure 存储。
  
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