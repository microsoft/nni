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

* **debug**
  
  * 说明
    
    NNI will check the version of nniManager process and the version of trialKeeper in remote, pai and kubernetes platform. If you want to disable version check, you could set debug be false.

* **maxTrialNum**
  
  * 说明
    
    **maxTrialNum** specifies the max number of trial jobs created by NNI, including succeeded and failed jobs.

* **trainingServicePlatform**
  
  * 说明
    
    **trainingServicePlatform** specifies the platform to run the experiment, including {**local**, **remote**, **pai**, **kubeflow**}.
    
    * **local** run an experiment on local ubuntu machine.
    
    * **remote** submit trial jobs to remote ubuntu machines, and **machineList** field should be filed in order to set up SSH connection to remote machine.
    
    * **pai** submit trial jobs to [OpenPai](https://github.com/Microsoft/pai) of Microsoft. For more details of pai configuration, please reference [PAIMOdeDoc](./PAIMode.md)
    
    * **kubeflow** submit trial jobs to [kubeflow](https://www.kubeflow.org/docs/about/kubeflow/), NNI support kubeflow based on normal kubernetes and [azure kubernetes](https://azure.microsoft.com/en-us/services/kubernetes-service/).

* **searchSpacePath**
  
  * 说明
    
    **searchSpacePath** specifies the path of search space file, which should be a valid path in the local linux machine.
    
    注意: 如果设置了 useAnnotation=True，searchSpacePath 字段必须被删除。

* **useAnnotation**
  
  * 说明
    
    **useAnnotation** use annotation to analysis trial code and generate search space.
    
    Note: if set useAnnotation=True, the searchSpacePath field should be removed.

* **nniManagerIp**
  
  * 说明
    
    **nniManagerIp** set the IP address of the machine on which NNI manager process runs. This field is optional, and if it's not set, eth0 device IP will be used instead.
    
    Note: run ifconfig on NNI manager's machine to check if eth0 device exists. If not, we recommend to set nnimanagerIp explicitly.

* **logDir**
  
  * 说明
    
    **logDir** configures the directory to store logs and data of the experiment. The default value is `<user home directory>/nni/experiment`

* **logLevel**
  
  * 说明
    
    **logLevel** sets log level for the experiment, available log levels are: `trace, debug, info, warning, error, fatal`. The default value is `info`.

* **tuner**
  
  * 说明
    
    **tuner** specifies the tuner algorithm in the experiment, there are two kinds of ways to set tuner. One way is to use tuner provided by NNI sdk, need to set **builtinTunerName** and **classArgs**. Another way is to use users' own tuner file, and need to set **codeDirectory**, **classFileName**, **className** and **classArgs**.
  
  * **builtinTunerName** and **classArgs**
    
    * **builtinTunerName**
      
      **builtinTunerName** specifies the name of system tuner, NNI sdk provides four kinds of tuner, including {**TPE**, **Random**, **Anneal**, **Evolution**, **BatchTuner**, **GridSearch**}
    
    * **classArgs**
      
      **classArgs** specifies the arguments of tuner algorithm. If the **builtinTunerName** is in {**TPE**, **Random**, **Anneal**, **Evolution**}, user should set **optimize_mode**.
  
  * **codeDir**, **classFileName**, **className** 和 **classArgs**
    
    * **codeDir**
      
      **codeDir** specifies the directory of tuner code.
    
    * **classFileName**
      
      **classFileName** specifies the name of tuner file.
    
    * **className**
      
      **className** specifies the name of tuner class.
    
    * **classArgs**
      
      **classArgs** specifies the arguments of tuner algorithm.
    
    * **gpuNum**
      
      **gpuNum** specifies the gpu number to run the tuner process. The value of this field should be a positive number.
      
      Note: users could only specify one way to set tuner, for example, set {tunerName, optimizationMode} or {tunerCommand, tunerCwd}, and could not set them both.

* **assessor**
  
  * Description
    
    **assessor** specifies the assessor algorithm to run an experiment, there are two kinds of ways to set assessor. One way is to use assessor provided by NNI sdk, users need to set **builtinAssessorName** and **classArgs**. Another way is to use users' own assessor file, and need to set **codeDirectory**, **classFileName**, **className** and **classArgs**.
  
  * **builtinAssessorName** and **classArgs**
    
    * **builtinAssessorName**
      
      **builtinAssessorName** specifies the name of system assessor, NNI sdk provides one kind of assessor {**Medianstop**}
    
    * **classArgs**
      
      **classArgs** specifies the arguments of assessor algorithm
  
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
    
    **command** 指定了运行 Trial 进程的命令行。
  
  * **codeDir**
    
    **codeDir** specifies the directory of your own trial file.
  
  * **gpuNum**
    
    **gpuNum** 指定了运行 Trial 进程的 GPU 数量。 默认值为 0。

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
      
      **vaultName** is the value of `--vault-name` used in az command.
    
    * **name**
      
      **name** is the value of `--name` used in az command.
  
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