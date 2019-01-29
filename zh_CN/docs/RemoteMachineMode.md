# Run an Experiment on Multiple Machines

NNI supports running an experiment on multiple machines through SSH channel, called `remote` mode. NNI assumes that you have access to those machines, and already setup the environment for running deep learning training code.

e.g. Three machines and you login in with account `bob` (Note: the account is not necessarily the same on different machine):

| IP       | 用户名 | 密码     |
| -------- | --- | ------ |
| 10.1.1.1 | bob | bob123 |
| 10.1.1.2 | bob | bob123 |
| 10.1.1.3 | bob | bob123 |

## 设置 NNI 环境

Install NNI on each of your machines following the install guide [here](GetStarted.md).

## 运行 Experiment

Install NNI on another machine which has network accessibility to those three machines above, or you can just use any machine above to run nnictl command line tool.

We use `examples/trials/mnist-annotation` as an example here. `cat ~/nni/examples/trials/mnist-annotation/config_remote.yml` to see the detailed configuration file:

```yml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#可选项: local, remote, pai
trainingServicePlatform: remote
#可选项: true, false
useAnnotation: true
tuner:
  #可选项: TPE, Random, Anneal, Evolution, BatchTuner
  #SMAC (SMAC 需要通过 nnictl 安装)
  builtinTunerName: TPE
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
#local 模式下 machineList 可为空
machineList:
  - ip: 10.1.1.1
    username: bob
    passwd: bob123
    #使用默认端口 22 时，该配置可跳过
    #port: 22
  - ip: 10.1.1.2
    username: bob
    passwd: bob123
  - ip: 10.1.1.3
    username: bob
    passwd: bob123
```

Simply filling the `machineList` section and then run:

```bash
nnictl create --config ~/nni/examples/trials/mnist-annotation/config_remote.yml
```

to start the experiment.