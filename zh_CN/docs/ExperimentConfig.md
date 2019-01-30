# Experiment（实验）配置参考

创建 Experiment 时，需要给 nnictl 命令提供配置文件的路径。 配置文件是 YAML 格式，需要保证其格式正确。 本文介绍了配置文件的内容，并提供了一些示例和模板。

* [模板](#Template) (配置文件的模板)
* [配置说明](#Configuration) (配置文件每个项目的说明)
* [样例](#Examples) (配置文件样例)

<a name="Template"></a>

## 模板

* **简化版（不包含 Annotation（标记）和 Assessor）** 

    authorName: 
    experimentName: 
    trialConcurrency: 
    maxExecDuration: 
    maxTrialNum: 
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: 
    searchSpacePath: 
    #choice: true, false
    useAnnotation: 
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName:
      classArgs:
        #choice: maximize, minimize
        optimize_mode:
      gpuNum: 
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
    

* **使用 Assessor**

    authorName: 
    experimentName: 
    trialConcurrency: 
    maxExecDuration: 
    maxTrialNum: 
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: 
    searchSpacePath: 
    #choice: true, false
    useAnnotation: 
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName:
      classArgs:
        #choice: maximize, minimize
        optimize_mode:
      gpuNum: 
    assessor:
      #choice: Medianstop
      builtinAssessorName:
      classArgs:
        #choice: maximize, minimize
        optimize_mode:
      gpuNum: 
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
    

* **使用 Annotation**

    authorName: 
    experimentName: 
    trialConcurrency: 
    maxExecDuration: 
    maxTrialNum: 
    #choice: local, remote, pai, kubeflow
    trainingServicePlatform: 
    #choice: true, false
    useAnnotation: 
    tuner:
      #choice: TPE, Random, Anneal, Evolution
      builtinTunerName:
      classArgs:
        #choice: maximize, minimize
        optimize_mode:
      gpuNum: 
    assessor:
      #choice: Medianstop
      builtinAssessorName:
      classArgs:
        #choice: maximize, minimize
        optimize_mode:
      gpuNum: 
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
    
        __trialConcurrency__ specifies the max num of trial jobs run simultaneously.  
        
          Note: if trialGpuNum is bigger than the free gpu numbers, and the trial jobs running simultaneously can not reach trialConcurrency number, some trial jobs will be put into a queue to wait for gpu allocation.
        

* **maxExecDuration**
  
  * Description
  
  **maxExecDuration** specifies the max duration time of an experiment.The unit of the time is {**s**, **m**, **h**, **d**}, which means {*seconds*, *minutes*, *hours*, *days*}.
  
          Note: The maxExecDuration spec set the time of an experiment, not a trial job. If the experiment reach the max duration time, the experiment will not stop, but could not submit new trial jobs any more.
      

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
    
        __searchSpacePath__ specifies the path of search space file, which should be a valid path in the local linux machine.
        
        Note: if set useAnnotation=True, the searchSpacePath field should be removed.
        

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
    
    **builtinTunerName** specifies the name of system tuner, NNI sdk provides four kinds of tuner, including {**TPE**, **Random**, **Anneal**, **Evolution**, **BatchTuner**, **GridSearch**}
    
    * **classArgs**
      
      **classArgs** specifies the arguments of tuner algorithm. If the **builtinTunerName** is in {**TPE**, **Random**, **Anneal**, **Evolution**}, user should set **optimize_mode**.
  
  * **codeDir**, **classFileName**, **className** 和 **classArgs**
    
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
        

* **Assessor**
  
  * 说明
    
    **assessor** 指定了 Experiment 的 Assessor 算法。有两种方法可设置 Assessor。 一种方法是使用 SDK 提供的 Assessor，需要设置 **builtinAssessorName** 和 **classArgs**。 另一种方法，是使用用户自定义的 Assessor，需要设置 **codeDirectory**，**classFileName**，**className** 和 **classArgs**。
  
  * **builtinAssessorName** 和 **classArgs**
    
    * **builtinAssessorName**
      
          __builtinAssessorName__ specifies the name of system assessor, NNI sdk provides one kind of assessor {__Medianstop__}
          
    
    * **classArgs**
      
          __classArgs__ specifies the arguments of assessor algorithm
          
  
  * **codeDir**, **classFileName**, **className** 和 **classArgs**
    
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
    
        __command__  specifies the command to run trial process.
        
  
  * **codeDir**
    
        __codeDir__ specifies the directory of your own trial file.
        
  
  * **gpuNum**
    
        __gpuNum__ specifies the num of gpu to run the trial process. Default value is 0. 
        

* **trial (pai)**
  
  * **command**
    
        __command__  specifies the command to run trial process.
        
  
  * **codeDir**
    
        __codeDir__ specifies the directory of the own trial file.
        
  
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
      
      **iamge** 设置了 **ps** 使用的 docker 映像。
  
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
  
  Note: if users set passwd and sshKeyPath simultaneously, NNI will try passwd.
  
  * **passphrase**
    
    **passphrase** is used to protect ssh key, which could be empty if users don't have passphrase.

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
      
      **vaultName** is the value of ```--vault-name``` used in az command.
    
    * **name**
      
      **name** is the value of ```--name``` used in az command.
  
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
      gpuNum: 0
    trial:
      command: python3 mnist.py
      codeDir: /nni/mnist
      gpuNum: 0
    

    Could add assessor configuration in config file if set assessor.
    

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
    

    Or you could specify your own tuner and assessor file as following:
    

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
    

* **remote mode**

If run trial jobs in remote machine, users could specify the remote mahcine information as fllowing format:

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
    

* **pai mode**

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
    

* **kubeflow mode**

kubeflow use nfs as storage.

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
    

kubeflow use azure storage

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