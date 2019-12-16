# Run an Experiment on Multiple Machines

NNI supports running an experiment on multiple machines through SSH channel, called `remote` mode. NNI assumes that you have access to those machines, and already setup the environment for running deep learning training code.

e.g. Three machines and you login in with account `bob` (Note: the account is not necessarily the same on different machine):

| IP  | Username| Password |
| -------- |---------|-------|
| 10.1.1.1 | bob | bob123    |
| 10.1.1.2 | bob | bob123    |
| 10.1.1.3 | bob | bob123    |

## Setup NNI environment

Install NNI on each of your machines following the install guide [here](../Tutorial/QuickStart.md).

## Run an experiment

Install NNI on another machine which has network accessibility to those three machines above, or you can just run `nnictl` on any one of the three to launch the experiment.

We use `examples/trials/mnist-annotation` as an example here. Shown here is `examples/trials/mnist-annotation/config_remote.yml`:

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

Files in `codeDir` will be automatically uploaded to the remote machine. You can run NNI on different operating systems (Windows, Linux, MacOS) to spawn experiments on the remote machines (only Linux allowed):

```bash
nnictl create --config examples/trials/mnist-annotation/config_remote.yml
```

You can also use public/private key pairs instead of username/password for authentication. For advanced usages, please refer to [Experiment Config Reference](../Tutorial/ExperimentConfig.md).

## Version check

NNI support version check feature in since version 0.6, [reference](PaiMode.md).