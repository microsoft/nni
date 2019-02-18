# Experiment config reference

A config file is needed when create an experiment, the path of the config file is provide to nnictl. The config file is written in YAML format, and need to be written correctly. This document describes the rule to write config file, and will provide some examples and templates.

* [Template](#Template) (the templates of an config file)
* [Configuration spec](#Configuration) (the configuration specification of every attribute in config file)
* [Examples](#Examples) (the examples of config file)

<a name="Template"></a>

## Template

* **light weight(without Annotation and Assessor)**

```yaml
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
```

* **Use Assessor**

```yaml
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
```

* **Use Annotation**

```yaml
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
```

<a name="Configuration"></a>

## Configuration spec

* **authorName**
  
  * Description
    
    **authorName** is the name of the author who create the experiment. TBD: add default value

* **experimentName**
  
  * Description
    
    **experimentName** is the name of the experiment created.  
    TBD: add default value

* **trialConcurrency**
  
  * Description
    
    **trialConcurrency** specifies the max num of trial jobs run simultaneously.
    
    Note: if trialGpuNum is bigger than the free gpu numbers, and the trial jobs running simultaneously can not reach trialConcurrency number, some trial jobs will be put into a queue to wait for gpu allocation.

* **maxExecDuration**
  
  * Description
    
    **maxExecDuration** specifies the max duration time of an experiment.The unit of the time is {**s**, **m**, **h**, **d**}, which means {*seconds*, *minutes*, *hours*, *days*}.
    
    Note: The maxExecDuration spec set the time of an experiment, not a trial job. If the experiment reach the max duration time, the experiment will not stop, but could not submit new trial jobs any more.

* **maxTrialNum**
  
  * Description
    
    **maxTrialNum** specifies the max number of trial jobs created by NNI, including succeeded and failed jobs.

* **trainingServicePlatform**
  
  * Description
    
    **trainingServicePlatform** specifies the platform to run the experiment, including {**local**, **remote**, **pai**, **kubeflow**}.
    
    * **local** run an experiment on local ubuntu machine.
    
    * **remote** submit trial jobs to remote ubuntu machines, and **machineList** field should be filed in order to set up SSH connection to remote machine.
    
    * **pai** submit trial jobs to [OpenPai](https://github.com/Microsoft/pai) of Microsoft. For more details of pai configuration, please reference [PAIMOdeDoc](./PAIMode.md)
    
    * **kubeflow** submit trial jobs to [kubeflow](https://www.kubeflow.org/docs/about/kubeflow/), NNI support kubeflow based on normal kubernetes and [azure kubernetes](https://azure.microsoft.com/en-us/services/kubernetes-service/).

* **searchSpacePath**
  
  * Description
    
    **searchSpacePath** specifies the path of search space file, which should be a valid path in the local linux machine.
    
    Note: if set useAnnotation=True, the searchSpacePath field should be removed.

* **useAnnotation**
  
  * Description
    
    **useAnnotation** use annotation to analysis trial code and generate search space.
    
    Note: if set useAnnotation=True, the searchSpacePath field should be removed.

* **nniManagerIp**
  
  * Description
    
    **nniManagerIp** set the IP address of the machine on which NNI manager process runs. This field is optional, and if it's not set, eth0 device IP will be used instead.
    
    Note: run ifconfig on NNI manager's machine to check if eth0 device exists. If not, we recommend to set nnimanagerIp explicitly.

* **logDir**
  
  * Description
    
    **logDir** configures the directory to store logs and data of the experiment. The default value is `<user home directory>/nni/experiment`

* **logLevel**
  
  * Description
    
    **logLevel** sets log level for the experiment, available log levels are: `trace, debug, info, warning, error, fatal`. The default value is `info`.

* **tuner**
  
  * Description
    
    **tuner** specifies the tuner algorithm in the experiment, there are two kinds of ways to set tuner. One way is to use tuner provided by NNI sdk, need to set **builtinTunerName** and **classArgs**. Another way is to use users' own tuner file, and need to set **codeDirectory**, **classFileName**, **className** and **classArgs**.
  
  * **builtinTunerName** and **classArgs**
    
    * **builtinTunerName**
      
      **builtinTunerName** specifies the name of system tuner, NNI sdk provides four kinds of tuner, including {**TPE**, **Random**, **Anneal**, **Evolution**, **BatchTuner**, **GridSearch**}
    
    * **classArgs**
      
      **classArgs** specifies the arguments of tuner algorithm. If the **builtinTunerName** is in {**TPE**, **Random**, **Anneal**, **Evolution**}, user should set **optimize_mode**.
  
  * **codeDir**, **classFileName**, **className** and **classArgs**
    
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

## Examples

* **local mode**
  
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
    gpuNum: 0
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
  
  kubeflow with azure storage
  
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