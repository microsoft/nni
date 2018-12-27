**Run an Experiment on FrameworkController**
===
NNI supports running experiment using [FrameworkController](https://github.com/Microsoft/frameworkcontroller), called frameworkcontroller mode. FrameworkController is built to orchestrate all kinds of applications on Kubernetes, and you have to set a kubernetes cluster before using frameworkcontroller.

## Set up Kubernetes Service and kubeconfig
FrameworkController has same prerequisites as kubeflow mode except that you don't need to install kubeflow. Please refer the [document](./KubeflowMode.md) to set up your kubernetes cluster and other prerequisites for nni.

## Set up FrameworkController
Follow the [guideline](https://github.com/Microsoft/frameworkcontroller/tree/master/example/run) to set up the frameworkcontroller in the kubernetes cluster, nni support frameworkcontroller by the statefulset mode.

## Design
Please refer the design of [kubeflow training service](./KubeflowMode.md), frameworkcontroller training service pipeline is similar with kubeflow training service.

## Example

The frameworkcontroller config file format is:
```
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 10h
maxTrialNum: 100
#choice: local, remote, pai, kubeflow, frameworkcontroller
trainingServicePlatform: frameworkcontroller
searchSpacePath: ~/nni/examples/trials/mnist/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
#assessor:
#  builtinAssessorName: Medianstop
#  classArgs:
#    optimize_mode: maximize
#  gpuNum: 0
trial:
  codeDir: ~/nni/examples/trials/mnist
  taskRoles:
    - name: worker
      taskNum: 1
      command: python3 mnist.py
      gpuNum: 1
      cpuNum: 1
      memoryMB: 8192
      image: msranni/nni:latest
      frameworkAttemptCompletionPolicy:
        minFailedTaskCount: 1
        minSucceededTaskCount: 1
frameworkcontrollerConfig:
  storage: nfs
  nfs:
    server: {your_nfs_server}
    path: {your_nfs_server_exported_path}
```
If you use Azure Kubernetes Service, you should  set `kubeflowConfig` in your config yaml file as follows:
```
frameworkcontrollerConfig:
  storage: azureStorage
  keyVault:
    vaultName: {your_vault_name}
    name: {your_secert_name}
  azureStorage:
    accountName: {your_storage_account_name}
    azureShare: {your_azure_share_name}
```
Note: You should explicitly set `trainingServicePlatform: frameworkcontroller` in nni config yaml file if you want to start experiment in kubeflow mode. 

The trial's config format for nni frameworkcontroller mode is a simple version of frameworkcontroller's offical config, you could refer the [tensorflow example](https://github.com/Microsoft/frameworkcontroller/blob/master/example/framework/scenario/tensorflow/cpu/tensorflowdistributedtrainingwithcpu.yaml) for deeply understanding.  
Trial configuration in frameworkcontroller mode have the following configuration keys:
* taskRoles: you could set multiple task roles in config file, and each task role is a basic unit to process in kubernetes cluster.
   * name: the name of task role specified, like "worker", "ps", "master".
   * taskNum: the replica number of the task role.
   * command: the users' command to be used in the container.
   * gpuNum: the number of gpu device used in container.
   * cpuNum: the number of cpu device used in container.
   * memoryMB: the memory limitaion to be specified in container.
   * image: the docker image used to create pod and run the program.
   * frameworkAttemptCompletionPolicy: the policy to run framework, please refer the [user-manual](https://github.com/Microsoft/frameworkcontroller/blob/master/doc/user-manual.md) to get the specific information.

## How to run example
After you prepare a config file, you could run your experiment by nnictl. The way to start an experiment on frameworkcontroller is similar to kubeflow, please the [document](./KubeflowMode.md) for more information.