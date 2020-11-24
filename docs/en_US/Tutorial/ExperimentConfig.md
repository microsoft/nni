# Experiment Config Reference

A config file is needed when creating an experiment. The path of the config file is provided to `nnictl`.
The config file is in YAML format.
This document describes the rules to write the config file, and provides some examples and templates.

- [Experiment Config Reference](#experiment-config-reference)
  * [Template](#template)
  * [Configuration Spec](#configuration-spec)
    + [authorName](#authorname)
    + [experimentName](#experimentname)
    + [trialConcurrency](#trialconcurrency)
    + [maxExecDuration](#maxexecduration)
    + [versionCheck](#versioncheck)
    + [debug](#debug)
    + [maxTrialNum](#maxtrialnum)
    + [trainingServicePlatform](#trainingserviceplatform)
    + [searchSpacePath](#searchspacepath)
    + [useAnnotation](#useannotation)
    + [multiThread](#multithread)
    + [nniManagerIp](#nnimanagerip)
    + [logDir](#logdir)
    + [logLevel](#loglevel)
    + [logCollection](#logcollection)
    + [tuner](#tuner)
      - [builtinTunerName](#builtintunername)
      - [codeDir](#codedir)
      - [classFileName](#classfilename)
      - [className](#classname)
      - [classArgs](#classargs)
      - [gpuIndices](#gpuindices)
      - [includeIntermediateResults](#includeintermediateresults)
    + [assessor](#assessor)
      - [builtinAssessorName](#builtinassessorname)
      - [codeDir](#codedir-1)
      - [classFileName](#classfilename-1)
      - [className](#classname-1)
      - [classArgs](#classargs-1)
    + [advisor](#advisor)
      - [builtinAdvisorName](#builtinadvisorname)
      - [codeDir](#codedir-2)
      - [classFileName](#classfilename-2)
      - [className](#classname-2)
      - [classArgs](#classargs-2)
      - [gpuIndices](#gpuindices-1)
    + [trial](#trial)
    + [localConfig](#localconfig)
      - [gpuIndices](#gpuindices-2)
      - [maxTrialNumPerGpu](#maxtrialnumpergpu)
      - [useActiveGpu](#useactivegpu)
    + [machineList](#machinelist)
      - [ip](#ip)
      - [port](#port)
      - [username](#username)
      - [passwd](#passwd)
      - [sshKeyPath](#sshkeypath)
      - [passphrase](#passphrase)
      - [gpuIndices](#gpuindices-3)
      - [maxTrialNumPerGpu](#maxtrialnumpergpu-1)
      - [useActiveGpu](#useactivegpu-1)
      - [preCommand](#preCommand)
    + [kubeflowConfig](#kubeflowconfig)
      - [operator](#operator)
      - [storage](#storage)
      - [nfs](#nfs)
      - [keyVault](#keyvault)
      - [azureStorage](#azurestorage)
      - [uploadRetryCount](#uploadretrycount)
    + [paiConfig](#paiconfig)
      - [userName](#username)
      - [password](#password)
      - [token](#token)
      - [host](#host)
      - [reuse](#reuse)
  * [Examples](#examples)
    + [Local mode](#local-mode)
    + [Remote mode](#remote-mode)
    + [PAI mode](#pai-mode)
    + [Kubeflow mode](#kubeflow-mode)
    + [Kubeflow with azure storage](#kubeflow-with-azure-storage)

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

The name of the author who create the experiment.

*TBD: add default value.*

### experimentName

Required. String.

The name of the experiment created.

*TBD: add default value.*

### trialConcurrency

Required. Integer between 1 and 99999.

Specifies the max num of trial jobs run simultaneously.

If trialGpuNum is bigger than the free gpu numbers, and the trial jobs running simultaneously can not reach __trialConcurrency__ number, some trial jobs will be put into a queue to wait for gpu allocation.

### maxExecDuration

Optional. String. Default: 999d.

__maxExecDuration__ specifies the max duration time of an experiment. The unit of the time is {__s__, __m__, __h__, __d__}, which means {_seconds_, _minutes_, _hours_, _days_}.

Note: The maxExecDuration spec set the time of an experiment, not a trial job. If the experiment reach the max duration time, the experiment will not stop, but could not submit new trial jobs any more.

### versionCheck

Optional. Bool. Default: true.
  
NNI will check the version of nniManager process and the version of trialKeeper in remote, pai and kubernetes platform. If you want to disable version check, you could set versionCheck be false.

### debug

Optional. Bool. Default: false.

Debug mode will set versionCheck to false and set logLevel to be 'debug'.

### maxTrialNum

Optional. Integer between 1 and 99999. Default: 99999.

Specifies the max number of trial jobs created by NNI, including succeeded and failed jobs.

### trainingServicePlatform

Required. String.

Specifies the platform to run the experiment, including __local__, __remote__, __pai__, __kubeflow__, __frameworkcontroller__.

* __local__ run an experiment on local ubuntu machine.

* __remote__ submit trial jobs to remote ubuntu machines, and __machineList__ field should be filed in order to set up SSH connection to remote machine.

* __pai__  submit trial jobs to [OpenPAI](https://github.com/Microsoft/pai) of Microsoft. For more details of pai configuration, please refer to [Guide to PAI Mode](../TrainingService/PaiMode.md)

* __kubeflow__ submit trial jobs to [kubeflow](https://www.kubeflow.org/docs/about/kubeflow/), NNI support kubeflow based on normal kubernetes and [azure kubernetes](https://azure.microsoft.com/en-us/services/kubernetes-service/). For detail please refer to [Kubeflow Docs](../TrainingService/KubeflowMode.md)

* TODO: explain frameworkcontroller.

### searchSpacePath

Optional. Path to existing file.

Specifies the path of search space file, which should be a valid path in the local linux machine.

The only exception that __searchSpacePath__ can be not fulfilled is when `useAnnotation=True`.

### useAnnotation

Optional. Bool. Default: false.

Use annotation to analysis trial code and generate search space.

Note: if __useAnnotation__ is true, the searchSpacePath field should be removed.

### multiThread

Optional. Bool. Default: false.

Enable multi-thread mode for dispatcher. If multiThread is enabled, dispatcher will start a thread to process each command from NNI Manager.

### nniManagerIp

Optional. String. Default: eth0 device IP.

Set the IP address of the machine on which NNI manager process runs. This field is optional, and if it's not set, eth0 device IP will be used instead.

Note: run `ifconfig` on NNI manager's machine to check if eth0 device exists. If not, __nniManagerIp__ is recommended to set explicitly.

### logDir

Optional. Path to a directory. Default: `<user home directory>/nni-experiments`.

Configures the directory to store logs and data of the experiment.

### logLevel

Optional. String. Default: `info`.

Sets log level for the experiment. Available log levels are: `trace`, `debug`, `info`, `warning`, `error`, `fatal`.

### logCollection

Optional. `http` or `none`. Default: `none`.

Set the way to collect log in remote, pai, kubeflow, frameworkcontroller platform. There are two ways to collect log, one way is from `http`, trial keeper will post log content back from http request in this way, but this way may slow down the speed to process logs in trialKeeper. The other way is `none`, trial keeper will not post log content back, and only post job metrics. If your log content is too big, you could consider setting this param be `none`.

### tuner

Required.

Specifies the tuner algorithm in the experiment, there are two kinds of ways to set tuner. One way is to use tuner provided by NNI sdk (built-in tuners), in which case you need to set __builtinTunerName__ and __classArgs__. Another way is to use users' own tuner file, in which case __codeDirectory__, __classFileName__, __className__ and __classArgs__ are needed. *Users must choose exactly one way.*

#### builtinTunerName

Required if using built-in tuners. String.

Specifies the name of system tuner, NNI sdk provides different tuners introduced [here](../Tuner/BuiltinTuner.md).

#### codeDir

Required if using customized tuners. Path relative to the location of config file.

Specifies the directory of tuner code.

#### classFileName

Required if using customized tuners. File path relative to __codeDir__.

Specifies the name of tuner file.

#### className

Required if using customized tuners. String.

Specifies the name of tuner class.

#### classArgs

Optional. Key-value pairs. Default: empty.

Specifies the arguments of tuner algorithm. Please refer to [this file](../Tuner/BuiltinTuner.md) for the configurable arguments of each built-in tuner.

#### gpuIndices

Optional. String. Default: empty.

Specifies the GPUs that can be used by the tuner process. Single or multiple GPU indices can be specified. Multiple GPU indices are separated by comma `,`. For example, `1`, or `0,1,3`. If the field is not set, no GPU will be visible to tuner (by setting `CUDA_VISIBLE_DEVICES` to be an empty string).

#### includeIntermediateResults

Optional. Bool. Default: false.

If __includeIntermediateResults__ is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result.

### assessor

Specifies the assessor algorithm to run an experiment. Similar to tuners, there are two kinds of ways to set assessor. One way is to use assessor provided by NNI sdk. Users need to set __builtinAssessorName__ and __classArgs__. Another way is to use users' own assessor file, and users need to set __codeDirectory__, __classFileName__, __className__ and __classArgs__. *Users must choose exactly one way.*

By default, there is no assessor enabled.

#### builtinAssessorName

Required if using built-in assessors. String.

Specifies the name of built-in assessor, NNI sdk provides different assessors introduced [here](../Assessor/BuiltinAssessor.md).

#### codeDir

Required if using customized assessors. Path relative to the location of config file.

Specifies the directory of assessor code.

#### classFileName

Required if using customized assessors. File path relative to __codeDir__.

Specifies the name of assessor file.

#### className

Required if using customized assessors. String.

Specifies the name of assessor class.

#### classArgs

Optional. Key-value pairs. Default: empty.

Specifies the arguments of assessor algorithm.

### advisor

Optional.

Specifies the advisor algorithm in the experiment. Similar to tuners and assessors, there are two kinds of ways to specify advisor. One way is to use advisor provided by NNI sdk, need to set __builtinAdvisorName__ and __classArgs__. Another way is to use users' own advisor file, and need to set __codeDirectory__, __classFileName__, __className__ and __classArgs__.

When advisor is enabled, settings of tuners and advisors will be bypassed.

#### builtinAdvisorName

Specifies the name of a built-in advisor. NNI sdk provides [BOHB](../Tuner/BohbAdvisor.md) and [Hyperband](../Tuner/HyperbandAdvisor.md).

#### codeDir

Required if using customized advisors. Path relative to the location of config file.

Specifies the directory of advisor code.

#### classFileName

Required if using customized advisors. File path relative to __codeDir__.

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

* __command__: Required string. Specifies the command to run trial process.

* __codeDir__: Required string. Specifies the directory of your own trial file. This directory will be automatically uploaded in remote mode.

* __gpuNum__: Optional integer. Specifies the num of gpu to run the trial process. Default value is 0.

In PAI mode, the following keys are required.

* __command__: Required string. Specifies the command to run trial process.

* __codeDir__: Required string. Specifies the directory of the own trial file. Files in the directory will be uploaded in PAI mode.

* __gpuNum__: Required integer. Specifies the num of gpu to run the trial process. Default value is 0.

* __cpuNum__: Required integer. Specifies the cpu number of cpu to be used in pai container.

* __memoryMB__: Required integer. Set the memory size to be used in pai container, in megabytes.

* __image__: Required string. Set the image to be used in pai.

* __authFile__: Optional string. Used to provide Docker registry which needs authentication for image pull in PAI. [Reference](https://github.com/microsoft/pai/blob/2ea69b45faa018662bc164ed7733f6fdbb4c42b3/docs/faq.md#q-how-to-use-private-docker-registry-job-image-when-submitting-an-openpai-job).

* __shmMB__: Optional integer. Shared memory size of container.

* __portList__: List of key-values pairs with `label`, `beginAt`, `portNumber`. See [job tutorial of PAI](https://github.com/microsoft/pai/blob/master/docs/job_tutorial.md) for details.

In Kubeflow mode, the following keys are required.

* __codeDir__: The local directory where the code files are in.

* __ps__: An optional configuration for kubeflow's tensorflow-operator, which includes

    * __replicas__: The replica number of __ps__ role.

    * __command__: The run script in __ps__'s container.

    * __gpuNum__: The gpu number to be used in __ps__ container.

    * __cpuNum__: The cpu number to be used in __ps__ container.

    * __memoryMB__: The memory size of the container.

    * __image__: The image to be used in __ps__.

* __worker__: An optional configuration for kubeflow's tensorflow-operator.

    * __replicas__: The replica number of __worker__ role.

    * __command__: The run script in __worker__'s container.

    * __gpuNum__: The gpu number to be used in __worker__ container.

    * __cpuNum__: The cpu number to be used in __worker__ container.

    * __memoryMB__: The memory size of the container.

    * __image__: The image to be used in __worker__.

### localConfig

Optional in local mode. Key-value pairs.

Only applicable if __trainingServicePlatform__ is set to `local`, otherwise there should not be __localConfig__ section in configuration file.

#### gpuIndices

Optional. String. Default: none.

Used to specify designated GPU devices for NNI, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified. Multiple GPU indices should be separated with comma (`,`), such as `1` or  `0,1,3`. By default, all GPUs available will be used.

#### maxTrialNumPerGpu

Optional. Integer. Default: 1.
  
Used to specify the max concurrency trial number on a GPU device.
    
#### useActiveGpu

Optional. Bool. Default: false.

Used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no other active process in the GPU. If __useActiveGpu__ is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.

### machineList

Required in remote mode. A list of key-value pairs with the following keys.

#### ip

Required. IP address or host name that is accessible from the current machine.

The IP address or host name of remote machine.

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

If users use ssh key to login remote machine, __sshKeyPath__ should be a valid path to a ssh key file.

*Note: if users set passwd and sshKeyPath simultaneously, NNI will try passwd first.*

#### passphrase

Optional. String.

Used to protect ssh key, which could be empty if users don't have passphrase.

#### gpuIndices

Optional. String. Default: none.

Used to specify designated GPU devices for NNI, if it is set, only the specified GPU devices are used for NNI trial jobs. Single or multiple GPU indices can be specified. Multiple GPU indices should be separated with comma (`,`), such as `1` or  `0,1,3`. By default, all GPUs available will be used.

#### maxTrialNumPerGpu

Optional. Integer. Default: 1.

Used to specify the max concurrency trial number on a GPU device.

#### useActiveGpu

Optional. Bool. Default: false.

Used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no other active process in the GPU. If __useActiveGpu__ is set to true, NNI will use the GPU regardless of another processes. This field is not applicable for NNI on Windows.

#### preCommand

Optional. String.

Specifies the pre-command that will be executed before the remote machine executes other commands. Users can configure the experimental environment on remote machine by setting __preCommand__. If there are multiple commands need to execute, use `&&` to connect them, such as `preCommand: command1 && command2 && ...`.

__Note__: Because __preCommand__ will execute before other commands each time, it is strongly not recommended to set __preCommand__ that will make changes to system, i.e. `mkdir` or `touch`.

### remoteConfig

Optional field in remote mode. Users could set per machine information in `machineList` field, and set global configuration for remote mode in this field.

#### reuse

Optional. Bool. default: `false`. It's an experimental feature.

If it's true, NNI will reuse remote jobs to run as many as possible trials. It can save time of creating new jobs. User needs to make sure each trial can run independent in same job, for example, avoid loading checkpoint from previous trials. 

### kubeflowConfig

#### operator

Required. String. Has to be `tf-operator` or `pytorch-operator`.

Specifies the kubeflow's operator to be used, NNI support `tf-operator` in current version.

#### storage

Optional. String. Default. `nfs`.

Specifies the storage type of kubeflow, including `nfs` and `azureStorage`.

#### nfs

Required if using nfs. Key-value pairs.

* __server__ is the host of nfs server.

* __path__ is the mounted path of nfs.

#### keyVault

Required if using azure storage. Key-value pairs.

Set __keyVault__ to storage the private key of your azure storage account. Refer to https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2.

* __vaultName__ is the value of `--vault-name` used in az command.

* __name__ is the value of `--name` used in az command.

#### azureStorage

Required if using azure storage. Key-value pairs.

Set azure storage account to store code files.

* __accountName__ is the name of azure storage account.

* __azureShare__ is the share of the azure file storage.

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

#### reuse

Optional. Bool. default: `false`. It's an experimental feature.

If it's true, NNI will reuse OpenPAI jobs to run as many as possible trials. It can save time of creating new jobs. User needs to make sure each trial can run independent in same job, for example, avoid loading checkpoint from previous trials.

## Examples

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
      # Pre-command will be executed before the remote machine executes other commands.
      # Below is an example of specifying python environment.
      # If you want to execute multiple commands, please use "&&" to connect them.
      # preCommand: source ${replace_to_absolute_path_recommended_here}/bin/activate
      # preCommand: source ${replace_to_conda_path}/bin/activate ${replace_to_conda_env_name}
      preCommand: export PATH=${replace_to_python_environment_path_in_your_remote_machine}:$PATH
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
