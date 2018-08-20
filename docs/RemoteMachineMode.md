**Run an Experiment on Multiple Machines**
===
NNI supports running an experiment on multiple machines, called remote machine mode. Let's say you have multiple machines with the account `bob` (Note: the account is not necessarily the same on multiple machines):
| IP     | Username    | Password |
| --------|---------|-------|
| 10.1.1.1  | bob   | bob123    |
| 10.1.1.2 | bob | bob123    |
| 10.1.1.3 | bob | bob123    |

## Setup environment
Install NNI on each of your machines following the install guide [here](GetStarted.md).

## Run an experiment
Still using `examples/trials/mnist-annotation` as an example here. The yaml file you need is shown below: 
```
authorName: your_name
experimentName: auto_mnist
# how many trials could be concurrently running
trialConcurrency: 2
# maximum experiment running duration
maxExecDuration: 3h
# empty means never stop
maxTrialNum: 100
# choice: local, remote  
trainingServicePlatform: local
# choice: true, false  
useAnnotation: true
tuner:
  tunerName: TPE
  optimizationMode: Maximize
trial:
  trialCommand: python mnist.py
  trialCodeDir: /usr/share/nni/examples/trials/mnist-annotation
  trialGpuNum: 0
#machineList can be empty if the platform is local
machineList:
  - ip: 10.1.1.1
    username: bob
    passwd: bob123
  - ip: 10.1.1.2
    username: bob
    passwd: bob123
  - ip: 10.1.1.3
    username: bob
    passwd: bob123
```
Simply filling the `machineList` section. This yaml file is named `exp_remote.yaml`, then run:
```
nnictl create --config exp_remote.yaml
```
to start the experiment. This command can be executed on one of those three machines above, and can also be executed on another machine which has NNI installed and has network accessibility to those three machines.