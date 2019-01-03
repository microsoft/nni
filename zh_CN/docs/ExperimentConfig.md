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
    
        __trialConcurrency__ specifies the max num of trial jobs run simultaneously.  
        
        Note: if trialGpuNum is bigger than the free gpu numbers, and the trial jobs running simultaneously can not reach trialConcurrency number, some trial jobs will be put into a queue to wait for gpu allocation.
        

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
    
        __searchSpacePath__ specifies the path of search space file, which should be a valid path in the local linux machine.
        
        Note: if set useAnnotation=True, the searchSpacePath field should be removed.
        

* **useAnnotation**
  
  * 说明
    
    **useAnnotation** 定义使用标记来分析代码并生成搜索空间。
    
    Note: if set useAnnotation=True, the searchSpacePath field should be removed.

* **nniManagerIp**
  
  * 说明
    
    **nniManagerIp** 设置 NNI 管理器运行的 IP 地址。 此字段为可选项，如果没有设置，则会使用 eth0 的 IP 地址。
    
    Note: run ifconfig on NNI manager's machine to check if eth0 device exists. If not, we recommend to set nnimanagerIp explicitly.

* **tuner**
  
  * 说明
    
    **tuner** 指定了实验的调参器算法。有两种方法可设置调参器。 一种方法是使用 NNI SDK 提供的调参器，需要设置 **builtinTunerName** 和 **classArgs**。 另一种方法，是使用用户自定义的调参器，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  * **builtinTunerName** 和 **classArgs**
    
    * **builtinTunerName**
    
    **builtinTunerName** 指定了系统调参器的名字，NNI SDK 提供了多种调参器，如：{**TPE**, **Random**, **Anneal**, **Evolution**, **BatchTuner**, **GridSearch**}。
    
    * **classArgs**
      
      **classArgs** 指定了调参器算法的参数。 如果 **builtinTunerName** 是{**TPE**, **Random**, **Anneal**, **Evolution**}，用户需要设置 **optimize_mode**。
  
  * **codeDir**, **classFileName**, **className** and **classArgs**
    
    * **codeDir**
      
            __codeDir__ specifies the directory of tuner code.
          
    
    * **classFileName**
      
            __classFileName__ specifies the name of tuner file.
          
    
    * **className**
      
            __className__ specifies the name of tuner class.
          
    
    * **classArgs**
      
            __classArgs__ specifies the arguments of tuner algorithm.
          
  
  * **gpuNum**
    
        __gpuNum__ specifies the gpu number to run the tuner process. The value of this field should be a positive number.
        
        Note: users could only specify one way to set tuner, for example, set {tunerName, optimizationMode} or {tunerCommand, tunerCwd}, and could not set them both. 
        

* **assessor**
  
  * 说明
    
    **assessor** 指定了实验的评估器算法。有两种方法可设置评估器。 一种方法是使用 NNI SDK 提供的评估器，需要设置 **builtinAssessorName** 和 **classArgs**。 Another way is to use users' own assessor file, and need to set **codeDirectory**, **classFileName**, **className** and **classArgs**.
  
  * **builtinAssessorName** 和 **classArgs**
    
    * **builtinAssessorName**
      
          __builtinAssessorName__ specifies the name of system assessor, nni sdk provides one kind of assessor {__Medianstop__}
          
    
    * **classArgs**
      
          __classArgs__ specifies the arguments of assessor algorithm
          
  
  * **codeDir**, **classFileName**, **className** and **classArgs**
    
    * **codeDir**
      
           __codeDir__ specifies the directory of assessor code.
          
    
    * **classFileName**
      
           __classFileName__ specifies the name of assessor file.
          
    
    * **className**
      
           __className__ specifies the name of assessor class.
          
    
    * **classArgs**
      
           __classArgs__ specifies the arguments of assessor algorithm.
          
  
  * **gpuNum**
    
        __gpuNum__ specifies the gpu number to run the assessor process. The value of this field should be a positive number.
        
        Note: users' could only specify one way to set assessor, for example,set {assessorName, optimizationMode} or {assessorCommand, assessorCwd}, and users could not set them both.If users do not want to use assessor, assessor fileld should leave to empty. 
        

* **trial (local, remote)**
  
  * **command**
    
        __command__  指定了尝试进程的命令行。
        
  
  * **codeDir**
    
        __codeDir__ specifies the directory of your own trial file.
        
  
  * **gpuNum**
    
        __gpuNum__ specifies the num of gpu to run the trial process. Default value is 0. 
        

* **trial (pai)**
  
  * **command**
    
        __command__  指定了尝试进程的命令行。
        
  
  * **codeDir**
    
        __codeDir__ specifies the directory of the own trial file.
        
  
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
      
      **image** set the image to be used in **ps**.
  
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
      
      **image** set the image to be used in **worker**.

* **machineList**
  
       __machineList__ 要在 __trainingServicePlatform__=remote 时设置，否则其应为空。
      
  
  * **ip**
  
  **ip** 是远程计算机的 ip 地址。
  
  * **port**
  
  **端口** 是用于连接远程计算机的 ssh 端口。
  
  Note: if users set port empty, the default value will be 22.
  
  * **username**
  
  **username** is the account of remote machine.
  
  * **passwd**
  
  **passwd** specifies the password of the account.
  
  * **sshKeyPath**
    
    如果要使用 ssh 密钥登录远程计算机，则需要设置 **sshKeyPath**。 **sshKeyPath** 为有效的 ssh 密钥文件路径。
  
  Note: if users set passwd and sshKeyPath simultaneously, nni will try passwd.
  
  * **passphrase**
    
    **passphrase** is used to protect ssh key, which could be empty if users don't have passphrase.

* **kubeflowConfig**:
  
  * **operator**
    
    **operator** 指定了 kubeflow 使用的 operator，NNI 当前版本支持 **tf-operator**。
  
  * **storage**
    
    **storage** 指定了 kubeflow 的存储类型，包括 {**nfs**，**azureStorage**}。 此字段可选，默认值为 **nfs**。 如果使用了 azureStorage，此字段必须填写。
  
  * **nfs**
    
    **server** 是 NFS 服务器的地址
    
    **path** 是 NFS 挂载的路径
  
  * **keyVault**
    
    如果用户使用 Azure Kubernetes Service，需要设置 keyVault 来使用 Azure 存储账户的私钥。 参考: https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2
    
    * **vaultName**
      
      **vaultName** 是 az 命令中的 ```--vault-name``` 。
    
    * **name**
      
      **name** 是 az 命令中的 ```--name``` 。
  
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