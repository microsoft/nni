# 在多机上运行 Experiment

NNI 支持通过 SSH 通道在多台计算机上运行 Experiment，称为 `remote` 模式。 NNI 需要这些计算机的访问权限，并假定已配置好了深度学习训练环境。

例如：有三台服务器，登录账户为 `bob`（注意：账户不必在各台计算机上一致）：

| IP       | 用户名 | 密码     |
| -------- | --- | ------ |
| 10.1.1.1 | bob | bob123 |
| 10.1.1.2 | bob | bob123 |
| 10.1.1.3 | bob | bob123 |

## 设置 NNI 环境

按照[指南](QuickStart.md)在每台计算机上安装 NNI。

## 运行 Experiment

在另一台计算机，或在其中任何一台上安装 NNI，并运行 nnictl 工具。

以 `examples/trials/mnist-annotation` 为例。 `cat ~/nni/examples/trials/mnist-annotation/config_remote.yml` 来查看详细配置：

```yaml
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

可以使用不同系统来在远程计算机上运行 Experiment。

#### Linux 和 macOS

填好 `machineList` 部分，然后运行：

```bash
nnictl create --config ~/nni/examples/trials/mnist-annotation/config_remote.yml
```

来启动 Experiment。

#### Windows

填好 `machineList` 部分，然后运行：

```bash
nnictl create --config %userprofile%\nni\examples\trials\mnist-annotation\config_remote.yml
```

来启动 Experiment。

## 版本校验

从 0.6 开始，NNI 支持版本校验，详情参考[这里](PaiMode.md)。