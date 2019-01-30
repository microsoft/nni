# Experiment（实验）配置参考

创建 Experiment 时，需要给 nnictl 命令提供配置文件的路径。 配置文件是 YAML 格式，需要保证其格式正确。 本文介绍了配置文件的内容，并提供了一些示例和模板。

* [模板](#Template) (配置文件的模板)
* [配置说明](#Configuration) (配置文件每个项目的说明)
* [样例](#Examples) (配置文件样例)

<a name="Template"></a>

## 模板

* **简化版（不包含 Annotation（标记）和 Assessor）**

```yaml
```
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
```

* **使用 Assessor**

```yaml
```
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
```

* **使用 Annotation**

```yaml
```
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
    
        注意：如果 trialGpuNum 大于空闲的 GPU 数量，并且并发的尝试任务数量还没达到 trialConcurrency，尝试任务会被放入队列，等待分配 GPU 资源。
        

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
    
    **assessor** 指定了 Experiment 的 Assessor 算法。有两种方法可设置 Assessor。 One way is to use assessor provided by NNI sdk, users need to set **builtinAssessorName** and **classArgs**. Another way is to use users' own assessor file, and need to set **codeDirectory**, **classFileName**, **className** and **classArgs**.
  
  * **builtinAssessorName** and **classArgs**
    
    * **builtinAssessorName**
      
          __builtinAssessorName__ specifies the name of system assessor, NNI sdk provides one kind of assessor {__Medianstop__}
          
    
    * **classArgs**
      
          __classArgs__ specifies the arguments of assessor algorithm
          
  
  * **codeDir**, **classFileName**, **className** and **classArgs**
    
    * **codeDir**
      
      **codeDir** specifies the directory of assessor code.
    
    * **classFileName**
      
      **classFileName** specifies the name of assessor file.
    
    * **className**
      
      **className** specifies the name of assessor class.
    
    * **classArgs**
      
      **classArgs** specifies the arguments of assessor algorithm.
  
  * **gpuNum**
    
    **gpuNum** specifies the gpu number to run the assessor process. The value of this field should be a positive number.
    
    Note: users' could only specify one way to set assessor, for example,set {assessorName, optimizationMode} or {assessorCommand, assessorCwd}, and users could not set them both.If users do not want to use assessor, assessor fileld should leave to empty.

* **trial(local, remote)**
  
  * **command**
    
    **command** specifies the command to run trial process.
  
  * **codeDir**
    
    **codeDir** specifies the directory of your own trial file.
  
  * **gpuNum**
    
    **gpuNum** specifies the num of gpu to run the trial process. Default value is 0.

* **trial(pai)**
  
  * **command**
    
    **command** specifies the command to run trial process.
  
  * **codeDir**
    
    **codeDir** specifies the directory of the own trial file.
  
  * **gpuNum**
    
    **gpuNum** specifies the num of gpu to run the trial process. Default value is 0.
  
  * **cpuNum**
    
    **cpuNum** is the cpu number of cpu to be used in pai container.
  
  * **memoryMB**
    
    **memoryMB** set the momory size to be used in pai's container.
  
  * **image**
    
    **image** set the image to be used in pai.
  
  * **dataDir**
    
    **dataDir** is the data directory in hdfs to be used.
  
  * **outputDir**
    
    **outputDir** is the output directory in hdfs to be used in pai, the stdout and stderr files are stored in the directory after job finished.

* **trial(kubeflow)**
  
  * **codeDir**
    
    **codeDir** is the local directory where the code files in.
  
  * **ps(optional)**
    
    **ps** is the configuration for kubeflow's tensorflow-operator.
    
    * **replicas**
      
      **replicas** is the replica number of **ps** role.
    
    * **command**
      
      **command** is the run script in **ps**'s container.
    
    * **gpuNum**
      
      **gpuNum** set the gpu number to be used in **ps** container.
    
    * **cpuNum**
      
      **cpuNum** set the cpu number to be used in **ps** container.
    
    * **memoryMB**
      
      **memoryMB** set the memory size of the container.
    
    * **image**
      
      **image** set the image to be used in **ps**.
  
  * **worker**
    
    **worker** is the configuration for kubeflow's tensorflow-operator.
    
    * **replicas**
      
      **replicas** is the replica number of **worker** role.
    
    * **command**
      
      **command** is the run script in **worker**'s container.
    
    * **gpuNum**
      
      **gpuNum** set the gpu number to be used in **worker** container.
    
    * **cpuNum**
      
      **cpuNum** set the cpu number to be used in **worker** container.
    
    * **memoryMB**
      
      **memoryMB** set the memory size of the container.
    
    * **image**
      
      **image** set the image to be used in **worker**.

* **machineList**
  
  **machineList** should be set if **trainingServicePlatform** is set to remote, or it should be empty.
  
  * **ip**
    
    **ip** is the ip address of remote machine.
  
  * **port**
    
    **port** is the ssh port to be used to connect machine.
    
    Note: if users set port empty, the default value will be 22.
  
  * **username**
    
    **username** is the account of remote machine.
  
  * **passwd**
    
    **passwd** specifies the password of the account.
  
  * **sshKeyPath**
    
    If users use ssh key to login remote machine, could set **sshKeyPath** in config file. **sshKeyPath** is the path of ssh key file, which should be valid.
    
    Note: if users set passwd and sshKeyPath simultaneously, NNI will try passwd.
  
  * **passphrase**
    
    **passphrase** is used to protect ssh key, which could be empty if users don't have passphrase.

* **kubeflowConfig**:
  
  * **operator**
    
    **operator** specify the kubeflow's operator to be used, NNI support **tf-operator** in current version.
  
  * **storage**
    
    **storage** specify the storage type of kubeflow, including {**nfs**, **azureStorage**}. This field is optional, and the default value is **nfs**. If the config use azureStorage, this field must be completed.
  
  * **nfs**
    
    **server** is the host of nfs server
    
    **path** is the mounted path of nfs
  
  * **keyVault**
    
    If users want to use azure kubernetes service, they should set keyVault to storage the private key of your azure storage account. Refer: https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2
    
    * **vaultName**
      
      **vaultName** is the value of ```--vault-name``` used in az command.
    
    * **name**
      
      **name** is the value of ```--name``` used in az command.
  
  * **azureStorage**
    
    If users use azure kubernetes service, they should set azure storage account to store code files.
    
    * **accountName**
      
      **accountName** is the name of azure storage account.
    
    * **azureShare**
      
      **azureShare** is the share of the azure file storage.

* **paiConfig**
  
  * **userName**
    
    **userName** is the user name of your pai account.
  
  * **password**
    
    **password** is the password of the pai account.
  
  * **host**
    
    **host** is the host of pai.

<a name="Examples"></a>

## 样例

* **local mode**
  
  If users want to run trial jobs in local machine, and use annotation to generate search space, could use the following config:

```python
```
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

```

You can add assessor configuration.

```python
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
  gpuNum: 0
assessor:
  #choice: Medianstop
  builtinAssessorName: Medianstop
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
  gpuNum: 0
trial:
  command: python3 mnist.py
  codeDir: /nni/mnist
  gpuNum: 0
```

Or you could specify your own tuner and assessor file as following,

```python
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

* **remote mode**

If run trial jobs in remote machine, users could specify the remote mahcine information as fllowing format:

```python
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
  gpuNum: 0
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

* **pai mode**

```python
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
  #The hdfs directory to store data on pai, format 'hdfs://host:port/directory'
  dataDir: hdfs://10.11.12.13:9000/test
  #The hdfs directory to store output data generated by NNI, format 'hdfs://host:port/directory'
  outputDir: hdfs://10.11.12.13:9000/test
paiConfig:
  #The username to login pai
  userName: test
  #The password to login pai
  passWord: test
  #The host of restful server of pai
  host: 10.10.10.10
```

* **kubeflow mode**

kubeflow with nfs storage.

```python
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

kubeflow with azure storage

```python
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