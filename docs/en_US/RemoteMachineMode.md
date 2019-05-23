# Run an Experiment on Multiple Machines

NNI supports running an experiment on multiple machines through SSH channel, called `remote` mode. NNI assumes that you have access to those machines, and already setup the environment for running deep learning training code.

e.g. Three machines and you login in with account `bob` (Note: the account is not necessarily the same on different machine):

| IP  | Username| Password |
| -------- |---------|-------|
| 10.1.1.1 | bob | bob123    |
| 10.1.1.2 | bob | bob123    |
| 10.1.1.3 | bob | bob123    |

## Setup NNI environment

Install NNI on each of your machines following the install guide [here](QuickStart.md).

## Run an experiment

Install NNI on another machine which has network accessibility to those three machines above, or you can just use any machine above to run nnictl command line tool.

We use `examples/trials/mnist-annotation` as an example here. `cat ~/nni/examples/trials/mnist-annotation/config_remote.yml` to see the detailed configuration file:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: remote
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
You can use different systems to run experiments on the remote machine.
#### Linux and MacOS
Simply filling the `machineList` section and then run:

```bash
nnictl create --config ~/nni/examples/trials/mnist-annotation/config_remote.yml
```

to start the experiment.

#### Windows
Simply filling the `machineList` section and then run:

```bash
nnictl create --config %userprofile%\nni\examples\trials\mnist-annotation\config_remote.yml
```

to start the experiment.

## version check
NNI support version check feature in since version 0.6, [refer](PaiMode.md)