# Run an Experiment on Remote Machines

NNI can run one experiment on multiple remote machines through SSH, called `remote` mode. It's like a lightweight training platform. In this mode, NNI can be started from your computer, and dispatch trials to remote machines in parallel.

## Remote machine requirements

* It only supports Linux as remote machines, and [linux part in system specification](../Tutorial/Installation.md) is same as NNI local mode.

* Follow [installation](../Tutorial/Installation.md) to install NNI on each machine.

* Make sure remote machines meet environment requirements of your trial code. If the default environment does not meet the requirements, the setup script can be added into command parameter of NNI config.

* Make sure remote machines can be accessed through SSH from the machine which runs `nnictl` command. It supports both password and key authentication of SSH. For advanced usages, please refer to [machineList part of configuration](../Tutorial/ExperimentConfig.md).

* Make sure the NNI version on each machine is consistent.

## Run an experiment

e.g. there are three machines, which can be logged in with username and password.

| IP       | Username | Password |
| -------- | -------- | -------- |
| 10.1.1.1 | bob      | bob123   |
| 10.1.1.2 | bob      | bob123   |
| 10.1.1.3 | bob      | bob123   |

Install and run NNI on one of those three machines or another machine, which has network access to them.

Use `examples/trials/mnist-annotation` as the example. Below is content of `examples/trials/mnist-annotation/config_remote.yml`:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: remote
# search space file
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: true
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
#machineList can be empty if the platform is local
machineList:
  - ip: 10.1.1.1
    username: bob
    passwd: bob123
    #port can be skip if using default ssh port 22
    #port: 22
  - ip: 10.1.1.2
    username: bob
    passwd: bob123
  - ip: 10.1.1.3
    username: bob
    passwd: bob123
```

Files in `codeDir` will be uploaded to remote machines automatically. You can run below command on Windows, Linux, or macOS to spawn trials on remote Linux machines:

```bash
nnictl create --config examples/trials/mnist-annotation/config_remote.yml
```
