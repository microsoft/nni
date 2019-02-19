# Experiment（实验）配置参考

创建 Experiment 时，需要给 nnictl 命令提供配置文件的路径。 配置文件是 YAML 格式，需要保证其格式正确。 本文介绍了配置文件的内容，并提供了一些示例和模板。

* [模板](#Template) (配置文件的模板)
* [配置说明](#Configuration) (配置文件每个项目的说明)
* [样例](#Examples) (配置文件样例)

<a name="Template"></a>

## 模板

* **简化版（不包含 Annotation（标记）和 Assessor）**

```yaml
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#可选项: local, remote, pai, kubeflow
trainingServicePlatform: 
searchSpacePath: 
#可选项: true, false
useAnnotation: 
tuner:
  #可选项: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuNum: 
trial:
  command: 
  codeDir: 
  gpuNum: 
#在本地使用时，machineList 可为空
machineList:
  - ip: 
    port: 
    username: 
    passwd:
```

* **使用 Assessor**

```yaml
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#可选项: local, remote, pai, kubeflow
trainingServicePlatform: 
searchSpacePath: 
#可选项: true, false
useAnnotation: 
tuner:
  #可选项: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuNum: 
assessor:
  #可选项: Medianstop
  builtinAssessorName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuNum: 
trial:
  command: 
  codeDir: 
  gpuNum: 
#在本地使用时，machineList 可为空
machineList:
  - ip: 
    port: 
    username: 
    passwd:
```

* **使用 Annotation**

```yaml
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#可选项: local, remote, pai, kubeflow
trainingServicePlatform: 
#可选项: true, false
useAnnotation: 
tuner:
  #可选项: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuNum: 
assessor:
  #可选项: Medianstop
  builtinAssessorName:
  classArgs:
    #可选项: maximize, minimize
    optimize_mode:
  gpuNum: 
trial:
  command: 
  codeDir: 
  gpuNum: 
#在本地使用时，machineList 可为空
machineList:
  - ip: 
    port: 
    username: 
    passwd:
```

<a name="Configuration"></a>

## 说明

* **authorName**
  
  * 说明
    
    **authorName** 是创建 Experiment 的作者。 待定: 增加默认值

* **experimentName**
  
  * 说明
    
    **experimentName** 是 Experiment 的名称。  
    待实现：增加默认值

* **trialConcurrency**
  
  * 说明
    
    **trialConcurrency** 定义了并发尝试任务的最大数量。
    
    注意：如果 trialGpuNum 大于空闲的 GPU 数量，并且并发的 Trial 任务数量还没达到 trialConcurrency，Trial 任务会被放入队列，等待分配 GPU 资源。

* **maxExecDuration**
  
  * 说明
    
    **maxExecDuration** 定义 Experiment 执行的最长时间。时间单位：{**s**, **m**, **h**, **d**}，分别代表：{*seconds*, *minutes*, *hours*, *days*}。
    
    注意：maxExecDuration 设置的是 Experiment 执行的时间，不是 Trial 的。 如果 Experiment 达到了设置的最大时间，Experiment 不会停止，但不会再启动新的 Trial 作业。

* **maxTrialNum**
  
  * 说明
    
    **maxTrialNum** 定义了 Trial 任务的最大数量，成功和失败的都计算在内。

* **trainingServicePlatform**
  
  * 说明
    
    **trainingServicePlatform** 定义运行 Experiment 的平台，包括：{**local**, **remote**, **pai**, **kubeflow**}.
    
    * **local** 在本机的 ubuntu 上运行 Experiment。
    
    * **remote** 将任务提交到远程的 Ubuntu 上，必须用 **machineList** 来指定远程的 SSH 连接信息。
    
    * **pai** 提交任务到微软开源的 [OpenPAI](https://github.com/Microsoft/pai) 上。 更多 OpenPAI 配置，参考 [pai 模式](./PAIMode.md)。
    
    * **kubeflow** 提交任务至 [Kubeflow](https://www.kubeflow.org/docs/about/kubeflow/)。 NNI 支持基于 Kubeflow 的 Kubenetes，以及[Azure Kubernetes](https://azure.microsoft.com/en-us/services/kubernetes-service/)。

* **searchSpacePath**
  
  * 说明
    
    **searchSpacePath** 定义搜索空间文件的路径，此文件必须在运行 nnictl 的本机。
    
    注意: 如果设置了 useAnnotation=True，searchSpacePath 字段必须被删除。

* **useAnnotation**
  
  * 说明
    
    **useAnnotation** 定义使用标记来分析代码并生成搜索空间。
    
    注意: 如果设置了 useAnnotation=True，searchSpacePath 字段必须被删除。

* **nniManagerIp**
  
  * 说明
    
    **nniManagerIp** 设置 NNI 管理器运行的 IP 地址。 此字段为可选项，如果没有设置，则会使用 eth0 的 IP 地址。
    
    注意: 可在 NNI 管理器机器上运行 ifconfig 来检查 eth0 是否存在。 如果不存在，推荐显式设置 nnimanagerIp。

* **logDir**
  
  * 说明
    
    **logDir** 配置存储日志和数据的目录。 默认值是 `<user home directory>/nni/experiment`

* **logLevel**
  
  * 说明
    
    **logLevel** 为 Experiment 设置日志级别，支持的日志级别有：`trace, debug, info, warning, error, fatal`。 默认值是 `info`。

* **Tuner**
  
  * 说明
    
    **tuner** 指定了 Experiment 的 Tuner 算法。有两种方法可设置 Tuner。 一种方法是使用 SDK 提供的 Tuner，需要设置 **builtinTunerName** 和 **classArgs**。 另一种方法，是使用用户自定义的 Tuner，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  * **builtinTunerName** 和 **classArgs**
    
    * **builtinTunerName**
      
      **builtinTunerName** 指定了系统 Tuner 的名字，NNI SDK 提供了多种 Tuner，如：{**TPE**, **Random**, **Anneal**, **Evolution**, **BatchTuner**, **GridSearch**}。
    
    * **classArgs**
      
      **classArgs** 指定了 Tuner 算法的参数。 如果 **builtinTunerName** 是{**TPE**, **Random**, **Anneal**, **Evolution**}，用户需要设置 **optimize_mode**。
  
  * **codeDir**, **classFileName**, **className** 和 **classArgs**
    
    * **codeDir**
      
      **codeDir** 指定 Tuner 代码的目录。
    
    * **classFileName**
      
      **classFileName** 指定 Tuner 文件名。
    
    * **className**
      
      **className** 指定 Tuner 类名。
    
    * **classArgs**
      
      **classArgs** 指定了 Tuner 算法的参数。
    
    * **gpuNum**
      
      **gpuNum** 指定了运行 Tuner 进程的 GPU 数量。 此字段的值必须是正整数。
      
      注意: 只能使用一种方法来指定 Tuner，例如：设置{tunerName, optimizationMode} 或 {tunerCommand, tunerCwd}，不能同时设置。

* **Assessor**
  
  * 说明
    
    **assessor** 指定了 Experiment 的 Assessor 算法。有两种方法可设置 Assessor。 一种方法是使用 SDK 提供的 Assessor，需要设置 **builtinAssessorName** 和 **classArgs**。 另一种方法，是使用用户自定义的 Assessor，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  * **builtinAssessorName** 和 **classArgs**
    
    * **builtinAssessorName**
      
      **builtinAssessorName** 指定了系统 Assessor 的名称， NNI 内置的 Assessor 有 {**Medianstop**，等等}。
    
    * **classArgs**
      
      **classArgs** 指定了 Assessor 算法的参数。
  
  * **codeDir**, **classFileName**, **className** 和 **classArgs**
    
    * **codeDir**
      
      **codeDir** 指定 Assessor 代码的目录。
    
    * **classFileName**
      
      **classFileName** 指定 Assessor 文件名。
    
    * **className**
      
      **className** 指定 Assessor 类名。
    
    * **classArgs**
      
      **classArgs** 指定了 Assessor 算法的参数。
  
  * **gpuNum**
    
    **gpuNum** 指定了运行 Assessor 进程的 GPU 数量。 此字段的值必须是正整数。
    
    注意: 只能使用一种方法来指定 Assessor，例如：设置 {assessorName, optimizationMode} 或 {assessorCommand, assessorCwd}，不能同时设置。如果不需要使用 Assessor，可将其置为空。

* **trial (local, remote)**
  
  * **command**
    
    **command** 指定了运行 Trial 进程的命令行。
  
  * **codeDir**
    
    **codeDir** 指定了 Trial 代码文件的目录。
  
  * **gpuNum**
    
    **gpuNum** 指定了运行 Trial 进程的 GPU 数量。 默认值为 0。

* **trial (pai)**
  
  * **command**
    
    **command** 指定了运行 Trial 进程的命令行。
  
  * **codeDir**
    
    **codeDir** 指定了 Trial 代码文件的目录。
  
  * **gpuNum**
    
    **gpuNum** 指定了运行 Trial 进程的 GPU 数量。 默认值为 0。
  
  * **cpuNum**
    
    **cpuNum** 指定了 OpenPAI 容器中使用的 CPU 数量。
  
  * **memoryMB**
    
    **memoryMB** 指定了 OpenPAI 容器中使用的内存数量。
  
  * **image**
    
    **image** 指定了 OpenPAI 中使用的 docker 映像。
  
  * **dataDir**
    
    **dataDir** 是 HDFS 中用到的数据目录变量。
  
  * **outputDir**
    
    **outputDir** 是 HDFS 中用到的输出目录变量。在 OpenPAI 中，stdout 和 stderr 文件会在作业完成后，存放在此目录中。

* **trial (kubeflow)**
  
  * **codeDir**
    
    **codeDir** 指定了代码文件的本机路径。
  
  * **ps (可选)**
    
    **ps** 是 Kubeflow 的 Tensorflow-operator 配置。
    
    * **replicas**
      
      **replicas** 是 **ps** 角色的副本数量。
    
    * **command**
      
      **command** 是在 **ps** 的容器中运行的脚本命令。
    
    * **gpuNum**
      
      **gpuNum** 是在 **ps** 容器中使用的 GPU 数量。
    
    * **cpuNum**
      
      **cpuNum** 是在 **ps** 容器中使用的 CPU 数量。
    
    * **memoryMB**
      
      **memoryMB** 指定了容器中使用的内存数量。
    
    * **image**
      
      **image** 设置了 **ps** 使用的 docker 映像。
  
  * **worker**
    
    **worker** 是 Kubeflow 的 Tensorflow-operator 配置。
    
    * **replicas**
      
      **replicas** 是 **worker** 角色的副本数量。
    
    * **command**
      
      **command** 是在 **worker** 的容器中运行的脚本命令。
    
    * **gpuNum**
      
      **gpuNum** 是在 **worker** 容器中使用的 GPU 数量。
    
    * **cpuNum**
      
      **cpuNum** 是在 **worker** 容器中使用的 CPU 数量。
    
    * **memoryMB**
      
      **memoryMB** 指定了容器中使用的内存数量。
    
    * **image**
      
      **image** 设置了 **worker** 使用的 docker 映像。

* **machineList**
  
  如果 **trainingServicePlatform** 为 remote，则需要设置 **machineList**。否则应将其置为空。
  
  * **ip**
    
    **ip** 是远程计算机的 ip 地址。
  
  * **port**
    
    **端口** 是用于连接远程计算机的 ssh 端口。
    
    注意：如果 port 设为空，则为默认值 22。
  
  * **username**
    
    **username** 是远程计算机的用户名。
  
  * **passwd**
    
    **passwd** 指定了账户的密码。
  
  * **sshKeyPath**
    
    如果要使用 ssh 密钥登录远程计算机，则需要设置 **sshKeyPath**。 **sshKeyPath** 为有效的 ssh 密钥文件路径。
    
    注意：如果同时设置了 passwd 和 sshKeyPath，NNI 会使用 passwd。
  
  * **passphrase**
    
    **passphrase** 用于保护 ssh 密钥，如果没有使用，可为空。

* **kubeflowConfig**:
  
  * **operator**
    
    **operator** 指定了 kubeflow 使用的 operator，NNI 当前版本支持 **tf-operator**。
  
  * **存储**
    
    **storage** 指定了 kubeflow 的存储类型，包括 {**nfs**，**azureStorage**}。 此字段可选，默认值为 **nfs**。 如果使用了 azureStorage，此字段必须填写。
  
  * **nfs**
    
    **server** 是 NFS 服务器的地址
    
    **path** 是 NFS 挂载的路径
  
  * **keyVault**
    
    如果用户使用 Azure Kubernetes Service，需要设置 keyVault 来使用 Azure 存储账户的私钥。 参考: https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2
    
    * **vaultName**
      
      **vaultName** 是 az 命令中 `--vault-name` 的值。
    
    * **name**
      
      **name** 是 az 命令中 `--name` 的值。
  
  * **azureStorage**
    
    如果用户使用了 Azure Kubernetes Service，需要设置 Azure 存储账户来存放代码文件。
    
    * **accountName**
      
      **accountName** 是 Azure 存储账户的名称。
    
    * **azureShare**
      
      **azureShare** 是 Azure 文件存储的共享参数。

* **paiConfig**
  
  * **userName**
    
    **userName** 是 OpenPAI 的用户名。
  
  * **password**
    
    **password** 是 OpenPAI 用户的密码。
  
  * **host**
    
    **host** 是 OpenPAI 的主机地址。

<a name="Examples"></a>

## 样例

* **本机模式**
  
  如果要在本机运行 Trial 任务，并使用标记来生成搜索空间，可参考下列配置：
  
  ```yaml
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
    gpuNum: 0
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
    gpuNum: 0
  assessor:
    #可选项: Medianstop
    builtinAssessorName: Medianstop
    classArgs:
      #可选项: maximize, minimize
      optimize_mode: maximize
    gpuNum: 0
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
    gpuNum: 0
  assessor:
    codeDir: /nni/assessor
    classFileName: myassessor.py
    className: MyAssessor
    classArgs:
      #choice: maximize, minimize
      optimize_mode: maximize
    gpuNum: 0
  trial:
    command: python3 mnist.py
    codeDir: /nni/mnist
    gpuNum: 0
  ```

* **远程模式**
  
  如果在远程服务器上运行 Trial 任务，需要增加服务器信息：
  
  ```yaml
  authorName: test
  experimentName: test_experiment
  trialConcurrency: 3
  maxExecDuration: 1h
  maxTrialNum: 10
  #可选项: local, remote, pai, kubeflow
  trainingServicePlatform: remote
  searchSpacePath: /nni/search_space.json
  #可选项: true, false
  useAnnotation: false
  tuner:
    #可选项: TPE, Random, Anneal, Evolution
    builtinTunerName: TPE
    classArgs:
      #可选项: maximize, minimize
      optimize_mode: maximize
    gpuNum: 0
  trial:
    command: python3 mnist.py
    codeDir: /nni/mnist
    gpuNum: 0
  # 如果是本地 Experiment，machineList 可为空。
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

* **pai 模式**
  
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

* **Kubeflow 模式**
  
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
  #可选项: local, remote, pai, kubeflow
  trainingServicePlatform: kubeflow
  searchSpacePath: search_space.json
  #可选项: true, false
  useAnnotation: false
  #nniManagerIp: 10.10.10.10
  tuner:
    #可选项: TPE, Random, Anneal, Evolution
    builtinTunerName: TPE
    classArgs:
      #可选项: maximize, minimize
      optimize_mode: maximize
  assessor:
    builtinAssessorName: Medianstop
    classArgs:
      optimize_mode: maximize
    gpuNum: 0
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