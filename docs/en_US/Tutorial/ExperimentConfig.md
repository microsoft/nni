# Experiment Config Reference

A config file is needed when creating an experiment. The path of the config file is provided to `nnictl`.
The config file is in YAML format.
This document describes the rules to write the config file, and provides some examples and templates.

- [Experiment Config Reference](#experiment-config-reference)
  - [Template](#template)
  - [Configuration Spec](#configuration-spec)
  - [Examples](#examples)

## Template

* __Light weight (without Annotation and Assessor)__

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

* __Use Assessor__

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

* __Use Annotation__

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

## Configuration Spec

### authorName

Required. String.

__authorName__ is the name of the author who create the experiment.

*TBD: add default value.*

### experimentName

Required. String.

__experimentName__ is the name of the experiment created.

*TBD: add default value.*

### trialConcurrency

Required. Integer between 1 and 99999.

__trialConcurrency__ specifies the max num of trial jobs run simultaneously.

If trialGpuNum is bigger than the free gpu numbers, and the trial jobs running simultaneously can not reach __trialConcurrency__ number, some trial jobs will be put into a queue to wait for gpu allocation.

### maxExecDuration

Optional. String. Default: 999d.

__maxExecDuration__ specifies the max duration time of an experiment. The unit of the time is {__s__, __m__, __h__, __d__}, which means {_seconds_, _minutes_, _hours_, _days_}.

Note: The maxExecDuration spec set the time of an experiment, not a trial job. If the experiment reach the max duration time, the experiment will not stop, but could not submit new trial jobs any more.

### versionCheck

Optional. Bool. Default: false.
  
NNI will check the version of nniManager process and the version of trialKeeper in remote, pai and kubernetes platform. If you want to disable version check, you could set versionCheck be false.

### debug

Optional. Bool. Default: false.

Debug mode will set versionCheck to false and set logLevel to be 'debug'.

### maxTrialNum

Optional. Integer between 1 and 99999. Default: 99999.

__maxTrialNum__ specifies the max number of trial jobs created by NNI, including succeeded and failed jobs.

### trainingServicePlatform

Required. String.

__trainingServicePlatform__ specifies the platform to run the experiment, including __local__, __remote__, __pai__, __kubeflow__, __frameworkcontroller__.

* __local__ run an experiment on local ubuntu machine.

* __remote__ submit trial jobs to remote ubuntu machines, and __machineList__ field should be filed in order to set up SSH connection to remote machine.

* __pai__  submit trial jobs to [OpenPAI](https://github.com/Microsoft/pai) of Microsoft. For more details of pai configuration, please refer to [Guide to PAI Mode](../TrainingService/PaiMode.md)

* __kubeflow__ submit trial jobs to [kubeflow](https://www.kubeflow.org/docs/about/kubeflow/), NNI support kubeflow based on normal kubernetes and [azure kubernetes](https://azure.microsoft.com/en-us/services/kubernetes-service/). For detail please refer to [Kubeflow Docs](../TrainingService/KubeflowMode.md)

* TODO: explain frameworkcontroller.

### searchSpacePath

Optional. Path to existing file.

__searchSpacePath__ specifies the path of search space file, which should be a valid path in the local linux machine.

The only exception that __searchSpacePath__ can be not fulfilled is when `useAnnotation=True`.

### useAnnotation

Optional. Bool. Default: false.

__useAnnotation__ use annotation to analysis trial code and generate search space.

Note: if __useAnnotation__ is true, the searchSpacePath field should be removed.

### multiPhase

Optional. Bool. Default: false.

__multiPhase__ enable [multi-phase experiment](../AdvancedFeature/MultiPhase.md).

### multiThread

Optional. Bool. Default: false.

__multiThread__ enable multi-thread mode for dispatcher, if multiThread is set to `true`, dispatcher will start a thread to process each command from NNI Manager.

### nniManagerIp

Optional. String. Default: eth0 device IP.

__nniManagerIp__ set the IP address of the machine on which NNI manager process runs. This field is optional, and if it's not set, eth0 device IP will be used instead.

Note: run `ifconfig` on NNI manager's machine to check if eth0 device exists. If not, __nniManagerIp__ is recommended to set explicitly.

### logDir

Optional. Path to a directory. Default: `<user home directory>/nni/experiment`.

__logDir__ configures the directory to store logs and data of the experiment.

### logLevel

Optional. String. Default: `info`.

__logLevel__ sets log level for the experiment. Available log levels are: `trace`, `debug`, `info`, `warning`, `error`, `fatal`.

### logCollection

__logCollection__ set the way to collect log in remote, pai, kubeflow, frameworkcontroller platform. There are two ways to collect log, one way is from `http`, trial keeper will post log content back from http request in this way, but this way may slow down the speed to process logs in trialKeeper. The other way is `none`, trial keeper will not post log content back, and only post job metrics. If your log content is too big, you could consider setting this param be `none`.

### tuner

__tuner__ specifies the tuner algorithm in the experiment, there are two kinds of ways to set tuner. One way is to use tuner provided by NNI sdk (built-in tuners), in which case you need to set __builtinTunerName__ and __classArgs__. Another way is to use users' own tuner file, in which case __codeDirectory__, __classFileName__, __className__ and __classArgs__ are needed. *Users must choose exactly one way.*

#### builtinTunerName

Required if using built-in tuners. String.

__builtinTunerName__ specifies the name of system tuner, NNI sdk provides different tuners introduced [here](../Tuner/BuiltinTuner.md).

#### codeDir

Required if using customized tuners. Path relative to the location of config file.

__codeDir__ specifies the directory of tuner code.

#### classFileName

Required if using customized tuners. File path relative to __codeDir__.

__classFileName__ specifies the name of tuner file.

#### classArgs

Optional. Key-value pairs. Default: empty.

__classArgs__ specifies the arguments of tuner algorithm. 

__classArgs__ specifies the arguments of tuner algorithm. Please refer to [this file](../Tuner/BuiltinTuner.md) for the configurable arguments of each built-in tuner.

#### gpuIndices

Optional. String. Default: empty.

__gpuIndices__ specifies the GPUs that can be used by the tuner process. Single or multiple GPU indices can be specified. Multiple GPU indices are separated by comma `,`. For example, `1`, or `0,1,3`. If the field is not set, no GPU will be visible to tuner (by setting `CUDA_VISIBLE_DEVICES` to be an empty string).

#### includeIntermediateResults

Optional. Bool. Default: false.

If __includeIntermediateResults__ is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result.

### assessor

__assessor__ specifies the assessor algorithm to run an experiment. Similar to tuners, there are two kinds of ways to set assessor. One way is to use assessor provided by NNI sdk. Users need to set __builtinAssessorName__ and __classArgs__. Another way is to use users' own assessor file, and users need to set __codeDirectory__, __classFileName__, __className__ and __classArgs__. *Users must choose exactly one way.*

  * __builtinAssessorName__ and __classArgs__
    * __builtinAssessorName__

      __builtinAssessorName__ specifies the name of built-in assessor, NNI sdk provides different assessors introducted [here](../Assessor/BuiltinAssessor.md).
    * __classArgs__

      __classArgs__ specifies the arguments of assessor algorithm

  * __codeDir__, __classFileName__, __className__ and __classArgs__

    * __codeDir__

      __codeDir__ specifies the directory of assessor code.

    * __classFileName__

      __classFileName__ specifies the name of assessor file.

    * __className__

      __className__ specifies the name of assessor class.

    * __classArgs__

      __classArgs__ specifies the arguments of assessor algorithm.

  Note: users could only use one way to specify assessor, either specifying `builtinAssessorName` and `classArgs`, or specifying `codeDir`, `classFileName`, `className` and `classArgs`. If users do not want to use assessor, assessor fileld should leave to empty.

### advisor
  * Description

    __advisor__ specifies the advisor algorithm in the experiment, there are two kinds of ways to specify advisor. One way is to use advisor provided by NNI sdk, need to set __builtinAdvisorName__ and __classArgs__. Another way is to use users' own advisor file, and need to set __codeDirectory__, __classFileName__, __className__ and __classArgs__.
  * __builtinAdvisorName__ and __classArgs__
    * __builtinAdvisorName__

      __builtinAdvisorName__ specifies the name of a built-in advisor, NNI sdk provides [different advisors](../Tuner/BuiltinTuner.md).

    * __classArgs__

      __classArgs__ specifies the arguments of the advisor algorithm. Please refer to [this file](../Tuner/BuiltinTuner.md) for the configurable arguments of each built-in advisor.
  * __codeDir__, __classFileName__, __className__ and __classArgs__
    * __codeDir__

      __codeDir__ specifies the directory of advisor code.
    * __classFileName__

      __classFileName__ specifies the name of advisor file.
    * __className__

      __className__ specifies the name of advisor class.
    * __classArgs__

      __classArgs__ specifies the arguments of advisor algorithm.

  * __gpuIndices__

      __gpuIndices__ specifies the gpus that can be used by the advisor process. Single or multiple GPU indices can be specified, multiple GPU indices are seperated by comma(,), such as `1` or `0,1,3`. If the field is not set, `CUDA_VISIBLE_DEVICES` will be '' in script, that is, no GPU is visible to tuner.

  Note: users could only use one way to specify advisor, either specifying `builtinAdvisorName` and `classArgs`, or specifying `codeDir`, `classFileName`, `className` and `classArgs`.

* __trial(local, remote)__

  * __command__

    __command__  specifies the command to run trial process.

  * __codeDir__

    __codeDir__ specifies the directory of your own trial file.

  * __gpuNum__

    __gpuNum__ specifies the num of gpu to run the trial process. Default value is 0.

* __trial(pai)__

  * __command__

    __command__  specifies the command to run trial process.

  * __codeDir__

    __codeDir__ specifies the directory of the own trial file.

  * __gpuNum__

    __gpuNum__ specifies the num of gpu to run the trial process. Default value is 0.

  * __cpuNum__

    __cpuNum__ is the cpu number of cpu to be used in pai container.

  * __memoryMB__

    __memoryMB__ set the momory size to be used in pai's container.

  * __image__

    __image__ set the image to be used in pai.

  * __dataDir__

    __dataDir__ is the data directory in hdfs to be used.

  * __outputDir__

    __outputDir__ is the output directory in hdfs to be used in pai, the stdout and stderr files are stored in the directory after job finished.

* __trial(kubeflow)__

  * __codeDir__

    __codeDir__ is the local directory where the code files in.

  * __ps(optional)__

    __ps__ is the configuration for kubeflow's tensorflow-operator.

    * __replicas__

      __replicas__ is the replica number of __ps__ role.

    * __command__

      __command__ is the run script in __ps__'s container.

    * __gpuNum__

      __gpuNum__ set the gpu number to be used in __ps__ container.

    * __cpuNum__

      __cpuNum__ set the cpu number to be used in __ps__ container.

    * __memoryMB__

      __memoryMB__ set the memory size of the container.

    * __image__

      __image__ set the image to be used in __ps__.

  * __worker__

    __worker__ is the configuration for kubeflow's tensorflow-operator.

    * __replicas__

      __replicas__ is the replica number of __worker__ role.

    * __command__

      __command__ is the run script in __worker__'s container.

    * __gpuNum__

      __gpuNum__ set the gpu number to be used in __worker__ container.

    * __cpuNum__

      __cpuNum__ set the cpu number to be used in __worker__ container.

    * __memoryMB__

      __memoryMB__ set the memory size of the container.

    * __image__

      __image__ set the image to be used in __worker__.

* __localConfig__

  __localConfig__ is applicable only if __trainingServicePlatform__ is set to `local`, otherwise there should not be __localConfig__ section in configuration file.
  * __gpuIndices__

    __gpuIndices__ is used to specify designated GPU devices for NNI, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified, multiple GPU indices are seperated by comma(,), such as `1` or  `0,1,3`.

  * __maxTrialNumPerGpu__
  
    __maxTrialNumPerGpu__ is used to specify the max concurrency trial number on a GPU device.
    
  * __useActiveGpu__
  
    __useActiveGpu__ is used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no another active process in the GPU, if __useActiveGpu__ is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.
  

* __machineList__

  __machineList__ should be set if __trainingServicePlatform__ is set to remote, or it should be empty.

  * __ip__

    __ip__ is the ip address of remote machine.

  * __port__

    __port__ is the ssh port to be used to connect machine.

     Note: if users set port empty, the default value will be 22.
  * __username__

    __username__ is the account of remote machine.
  * __passwd__

    __passwd__ specifies the password of the account.

  * __sshKeyPath__

    If users use ssh key to login remote machine, could set __sshKeyPath__ in config file. __sshKeyPath__ is the path of ssh key file, which should be valid.

    Note: if users set passwd and sshKeyPath simultaneously, NNI will try passwd.

  * __passphrase__

    __passphrase__ is used to protect ssh key, which could be empty if users don't have passphrase.

  * __gpuIndices__

    __gpuIndices__ is used to specify designated GPU devices for NNI on this remote machine, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified, multiple GPU indices are seperated by comma(,), such as `1` or  `0,1,3`.

  * __maxTrialNumPerGpu__
  
    __maxTrialNumPerGpu__ is used to specify the max concurrency trial number on a GPU device.

  * __useActiveGpu__
  
    __useActiveGpu__ is used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no another active process in the GPU, if __useActiveGpu__ is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.

* __kubeflowConfig__:

  * __operator__

    __operator__ specify the kubeflow's operator to be used, NNI support __tf-operator__ in current version.

  * __storage__

    __storage__ specify the storage type of kubeflow, including {__nfs__, __azureStorage__}. This field is optional, and the default value is __nfs__. If the config use azureStorage, this field must be completed.

  * __nfs__

    __server__ is the host of nfs server

    __path__ is the mounted path of nfs

  * __keyVault__

    If users want to use azure kubernetes service, they should set keyVault to storage the private key of your azure storage account. Refer: https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2

    * __vaultName__

      __vaultName__ is the value of `--vault-name` used in az command.

    * __name__

      __name__ is the value of `--name` used in az command.

  * __azureStorage__

    If users use azure kubernetes service, they should set azure storage account to store code files.

    * __accountName__

      __accountName__ is the name of azure storage account.

    * __azureShare__

      __azureShare__ is the share of the azure file storage.

  * __uploadRetryCount__

    If upload files to azure storage failed, NNI will retry the process of uploading, this field will specify the number of attempts to re-upload files.

* __paiConfig__

  * __userName__

    __userName__ is the user name of your pai account.

  * __password__

    __password__ is the password of the pai account.

  * __host__

    __host__ is the host of pai.

## Examples

* __local mode__

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

* __remote mode__

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

* __pai mode__

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

* __kubeflow mode__

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
