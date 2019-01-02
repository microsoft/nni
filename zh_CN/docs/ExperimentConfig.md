# 实验（Experiment）配置参考

创建实验时，需要给 nnictl 命令提供配置文件的路径。 配置文件是 yaml 格式，需要保证其格式正确。 本文介绍了配置文件的内容，并提供了一些示例和模板。

## 模板

* **简化版（不包含标记 (annotation) 和评估器）** 

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
    

* **使用评估器**

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
    

* **使用标记**

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
    

## 配置

* **authorName**
  
  * 说明
    
    **authorName**是创建实验的作者。 待定: 增加默认值

* **experimentName**
  
  * 说明
    
    **experimentName** 是实验的名称。  
    待实现：增加默认值

* **trialConcurrency**
  
  * 说明
    
    **trialConcurrency** 定义了并发尝试任务的最大数量。
    
        注意：如果 trialGpuNum 大于空闲的 GPU 数量，并且并发的尝试任务数量还没达到 trialConcurrency，尝试任务会被放入队列，等待分配 GPU 资源。
        

* **maxExecDuration**
  
  * 说明
  
  **maxExecDuration** 定义实验执行的最长时间。时间单位：{**s**, **m**, **h**, **d**}，分别代表：{*seconds*, *minutes*, *hours*, *days*}。

* **maxTrialNum**
  
  * 说明
    
    **maxTrialNum** 定义了尝试任务的最大数量，成功和失败的都计算在内。

* **trainingServicePlatform**
  
  * 说明
    
    **trainingServicePlatform** 定义运行实验的平台，包括：{**local**, **remote**, **pai**, **kubeflow**}.
    
    * **local** 在本机的 ubuntu 上运行实验。
    
    * **remote** 将任务提交到远程的 ubuntu 上，必须用 **machineList** 来指定远程的 SSH 连接信息。
    
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
        

* **tuner**
  
  * 说明
    
    **tuner** 指定了实验的调参器算法。有两种方法可设置调参器。 一种方法是使用 NNI SDK 提供的调参器，需要设置 **builtinTunerName** 和 **classArgs**。 另一种方法，是使用用户自定义的调参器，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  * **builtinTunerName** 和 **classArgs**
    
    * **builtinTunerName**
    
    **builtinTunerName** 指定了系统调参器的名字，NNI SDK 提供了多种调参器，如：{**TPE**, **Random**, **Anneal**, **Evolution**, **BatchTuner**, **GridSearch**}。
    
    * **classArgs**
      
      **classArgs** 指定了调参器算法的参数。 如果 **builtinTunerName** 是{**TPE**, **Random**, **Anneal**, **Evolution**}，用户需要设置 **optimize_mode**。
  
  * **codeDir**, **classFileName**, **className** 和**classArgs**  
    * **codeDir**
    
    **codeDir** 指定调参器代码的目录。
    
        * __classFileName__
        
    
    **classFileName** 指定调参器文件名。
    
    * **className**
    
    **className** 指定调参器类名。
    
    * **classArgs**
    
    **classArgs** 指定了调参器算法的参数。
  
  * **gpuNum**
    
    **gpuNum** 指定了运行调参器进程的 GPU 数量。 此字段的值必须是正整数。
    
        注意: 只能使用一种方法来指定调参器，例如：设置{tunerName, optimizationMode} 或 {tunerCommand, tunerCwd}，不能同时设置。 
        

* **assessor**
  
  * 说明
    
    **assessor** 指定了实验的评估器算法。有两种方法可设置评估器。 一种方法是使用 NNI SDK 提供的评估器，需要设置 **builtinAssessorName** 和 **classArgs**。 另一种方法，是使用用户自定义的评估器，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  * **builtinAssessorName** 和 **classArgs**
    
    * **builtinAssessorName**
    
    **builtinAssessorName** 指定了系统评估器的名字，NNI SDK 提供了一种评估器：{**Medianstop**}。
    
    * **classArgs**
      
      **classArgs** 指定了评估器算法的参数。
  
  * **codeDir**, **classFileName**, **className** and **classArgs**  
    * **codeDir**
    
    **codeDir** 指定评估器代码的目录。
    
        * __classFileName__
        
    
    **classFileName** 指定评估器文件名。
    
    * **className**
    
    **className** 指定评估器类名。
    
    * **classArgs**
    
    **classArgs** 指定了评估器算法的参数。
  
  * **gpuNum**
  
  **gpuNum** 指定了运行评估器进程的 GPU 数量。 此字段的值必须是正整数。
  
          注意: 只能使用一种方法来指定评估器，例如：设置 {assessorName, optimizationMode} 或 {assessorCommand, assessorCwd}，不能同时设置。如果不需要使用评估器，可将其置为空。 
      

* **trial (local, remote)**
  
  * **command**
    
        __command__  指定了尝试进程的命令行。
        
  
  * **codeDir**
    
    **codeDir** 指定了尝试代码文件的目录。
  
  * **gpuNum**
    
    **gpuNum** 指定了运行尝试进程的 GPU 数量。 默认值为 0。

* **trial (pai)**
  
  * **command**
    
        __command__  指定了尝试进程的命令行。
        
  
  * **codeDir**
    
    **codeDir** 指定了尝试代码文件的目录。
  
  * **gpuNum**
    
    **gpuNum** 指定了运行尝试进程的 GPU 数量。 默认值为 0。
  
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
      
      **iamge** set the image to be used in **ps**.
  
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
      
      **iamge** set the image to be used in **worker**.

* **machineList**
  
       __machineList__ should be set if users set __trainingServicePlatform__=remote, or it could be empty.
      
  
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
    
        Note: if users set passwd and sshKeyPath simultaneously, nni will try passwd.
        
  
  * **passphrase**
    
    **passphrase** is used to protect ssh key, which could be empty if users don't have passphrase.

* **kubeflowConfig**:
  
  * **operator**
    
    **operator** specify the kubeflow's operator to be used, nni support **tf-operator** in current version.
  
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

## 样例

* **本机模式**
  
  如果要在本机运行尝试任务，并使用标记来生成搜索空间，可参考下列配置：

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
    
    

    如果要设置评估器，可以增加评估器配置：
    

    ```
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
    

    或者可以指定自定义的调参器和评估器：
    

    ```
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

如果在远程服务器上运行尝试任务，需要增加服务器信息：

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
    # 如果是本地实验，machineList 可为空。
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
    
    

* **pai 模式**

    ```
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

Kubeflow 使用 NFS 作为存储。

    
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
    
    

Kubeflow 使用 Azure 存储

    
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